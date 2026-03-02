#!/usr/bin/env python3
"""ZMQ replay listener that drives a Realman robot arm.

Subscribes to rapid_driver's ZMQ PUB stream, extracts pose messages,
and sends them as waypoints to a RealmanInterpolationController.

Auto-discovers rapid_driver via mDNS or accepts a manual --zmq endpoint.

Usage:
    # Auto-discover rapid_driver, connect to arm:
    python replay_listener.py --robot_ip 192.168.0.204

    # Manual ZMQ endpoint:
    python replay_listener.py --zmq tcp://192.168.0.100:5560 --robot_ip 192.168.0.204

    # Dry-run (no robot, just print poses):
    python replay_listener.py --dry_run
"""

import argparse
import json
import signal
import sys
import time
from collections import defaultdict
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from iphone_replay import REMAP_PRESETS, mat_to_pose, pose_to_mat
from mcap_pose_loader import _pose_payload_to_transform


# ─── mDNS auto-discovery ────────────────────────────────────────────────────

def discover_rapiddriver(timeout: float = 5.0):
    """Discover rapid_driver via mDNS, return (host_ip, zmq_port)."""
    from zeroconf import ServiceBrowser, Zeroconf

    SERVICE_TYPE = "_rapiddriver._tcp.local."
    result = {}

    class Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if info is None:
                return
            addresses = info.parsed_scoped_addresses()
            if not addresses:
                return
            host = addresses[0]
            zmq_port = None
            if info.properties:
                zmq_port_bytes = info.properties.get(b"zmq_port")
                if zmq_port_bytes:
                    zmq_port = int(zmq_port_bytes.decode())
            result["host"] = host
            result["zmq_port"] = zmq_port or 5560

        def remove_service(self, zc, type_, name):
            pass

        def update_service(self, zc, type_, name):
            pass

    zc = Zeroconf()
    listener = Listener()
    browser = ServiceBrowser(zc, SERVICE_TYPE, listener)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "host" in result:
            break
        time.sleep(0.1)

    zc.close()

    if "host" not in result:
        return None
    return result["host"], result["zmq_port"]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ZMQ replay listener for Realman robot arm")
    parser.add_argument(
        "--zmq", type=str, default=None,
        help="ZMQ endpoint (e.g. tcp://192.168.0.100:5560). Auto-discovered if omitted.")
    parser.add_argument(
        "--robot_ip", "-ri", type=str, default="10.90.0.210",
        help="Realman robot IP (default: 10.90.0.210)")
    parser.add_argument(
        "--remap", default="realman",
        choices=list(REMAP_PRESETS.keys()),
        help="Coordinate remap preset (default: realman)")
    parser.add_argument(
        "--max_velocity", type=float, default=0.25,
        help="Safety speed limit m/s (default: 0.25)")
    parser.add_argument(
        "--frequency", "-f", type=float, default=100.0,
        help="Robot controller frequency Hz (default: 100)")
    parser.add_argument(
        "--follow", action=argparse.BooleanOptionalAction,
        default=False, help="Realman follow mode")
    parser.add_argument(
        "--lookahead", type=float, default=0.0,
        help="Command-side smoothing seconds (default: 0)")
    parser.add_argument(
        "--pose_topic", type=str, default="/pose",
        help="Topic substring filter for pose messages (default: /pose)")
    parser.add_argument(
        "--session_gap_secs", type=float, default=2.0,
        help="Wall-time gap to detect new session and reset anchor (default: 2.0)")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print poses without connecting to robot")
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="mDNS discovery timeout in seconds (default: 5)")
    args = parser.parse_args()

    # ── 1. Resolve ZMQ endpoint ──────────────────────────────────────────
    if args.zmq:
        endpoint = args.zmq
        print(f"Using manual ZMQ endpoint: {endpoint}")
    else:
        print(f"Discovering rapid_driver via mDNS (timeout={args.timeout}s)...")
        result = discover_rapiddriver(timeout=args.timeout)
        if result is None:
            print("ERROR: Could not discover rapid_driver via mDNS.", file=sys.stderr)
            print("Use --zmq tcp://<ip>:5560 to connect manually.", file=sys.stderr)
            sys.exit(1)
        host, zmq_port = result
        endpoint = f"tcp://{host}:{zmq_port}"
        print(f"Discovered rapid_driver at {endpoint}")

    # ── 2. Connect ZMQ SUB ───────────────────────────────────────────────
    import zmq
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
    print(f"Connected to {endpoint}, waiting for messages...")

    # ── 3. Robot setup ───────────────────────────────────────────────────
    robot = None
    shm_manager = None
    anchor_T = None       # 4x4 anchor transform (robot EEF at first pose)
    first_pose_T = None   # 4x4 first MCAP pose (for computing relative)
    last_msg_wall = None  # wall-clock of last received pose
    frame_count = 0
    topic_counts = defaultdict(int)

    R_remap = REMAP_PRESETS[args.remap]
    use_remap = not np.allclose(R_remap, np.eye(3))
    if use_remap:
        R_hom = np.eye(4)
        R_hom[:3, :3] = R_remap
        R_hom_inv = np.linalg.inv(R_hom)

    if not args.dry_run:
        from diffusion_policy.real_world.realman_interpolation_controller import (
            RealmanInterpolationController,
        )

        shm_manager = SharedMemoryManager()
        shm_manager.start()

        robot = RealmanInterpolationController(
            shm_manager=shm_manager,
            robot_ip=args.robot_ip,
            frequency=args.frequency,
            max_pos_speed=args.max_velocity * 1.73,
            max_rot_speed=0.5,
            joints_init=None,
            follow=args.follow,
            lookahead_time=args.lookahead,
        )
        robot.start()
        time.sleep(0.5)
        print(f"Robot connected at {args.robot_ip}, holding current pose.")
    else:
        print("Dry-run mode: no robot connection.")

    def cleanup():
        if robot is not None:
            try:
                robot.slow_stop()
            except Exception:
                robot.stop()
        if shm_manager is not None:
            shm_manager.shutdown()

    def signal_handler(sig, frame):
        elapsed = time.time() - t_start if t_start else 0
        print(f"\n{'='*50}")
        print(f"  Statistics ({elapsed:.1f}s)")
        print(f"{'='*50}")
        print(f"  Pose frames processed: {frame_count}")
        for topic in sorted(topic_counts):
            print(f"    {topic}: {topic_counts[topic]}")
        print(f"{'='*50}")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    t_start = time.time()
    print(f"\nListening for pose messages (topic filter: '{args.pose_topic}')...\n")

    # ── 4. Main loop ─────────────────────────────────────────────────────
    while True:
        if not sub.poll(100):
            continue

        frames = sub.recv_multipart()
        if len(frames) >= 2:
            topic = frames[0].decode("utf-8", errors="replace")
            payload_bytes = frames[1]
        elif len(frames) == 1:
            topic = "(none)"
            payload_bytes = frames[0]
        else:
            continue

        topic_counts[topic] += 1

        # ── 4b. Handle /replay/status control frames ─────────────────
        if topic == "/replay/status":
            try:
                event = json.loads(payload_bytes)
                if event.get("event") == "start":
                    sid = event.get("session_id", "?")
                    secs = event.get("total_secs", 0)
                    msgs = event.get("message_count", 0)
                    spd = event.get("speed", 1.0)
                    print(f"\n  === New session: {sid} ({secs:.1f}s, {msgs} msgs, speed={spd:.1f}x) ===")
                    anchor_T = None
                    first_pose_T = None
                    frame_count = 0
                    last_msg_wall = None
                elif event.get("event") == "stop":
                    sid = event.get("session_id", "?")
                    print(f"\n  === Session stopped: {sid} ({frame_count} frames) ===")
            except Exception as e:
                print(f"  [WARN] Failed to parse control frame: {e}")
            continue

        # ── 5. Topic filter ──────────────────────────────────────────
        if args.pose_topic not in topic:
            continue

        # ── 6. Parse pose ────────────────────────────────────────────
        try:
            payload = json.loads(payload_bytes)
            T_msg = _pose_payload_to_transform(payload)
        except Exception as e:
            print(f"  [WARN] Failed to parse pose: {e}")
            continue

        now = time.time()

        # ── 7. Session gap detection → reset anchor ──────────────────
        if last_msg_wall is not None and (now - last_msg_wall) > args.session_gap_secs:
            print(f"\n  Session gap detected ({now - last_msg_wall:.1f}s > {args.session_gap_secs}s), resetting anchor.")
            anchor_T = None
            first_pose_T = None
            frame_count = 0

        last_msg_wall = now

        # ── 8. First pose → record anchor ────────────────────────────
        if first_pose_T is None:
            first_pose_T = T_msg.copy()
            if robot is not None:
                state = robot.get_state()
                anchor_pose = state["ActualTCPPose"]
                anchor_T = pose_to_mat(np.array(anchor_pose))
            else:
                anchor_T = np.eye(4)
            print(f"  Anchor set. First pose recorded.")
            frame_count += 1
            continue

        # ── 9. Compute relative transform ────────────────────────────
        T_rel = np.linalg.inv(first_pose_T) @ T_msg

        # ── 10. Apply remap ──────────────────────────────────────────
        if use_remap:
            T_rel = R_hom @ T_rel @ R_hom_inv

        # ── 11. Compute absolute target ──────────────────────────────
        T_abs = anchor_T @ T_rel
        target_pose = mat_to_pose(T_abs)

        frame_count += 1

        if args.dry_run:
            if frame_count % 50 == 0 or frame_count == 1:
                pos = target_pose[:3]
                print(f"  [{frame_count:>6}] pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        else:
            # Schedule waypoint slightly in the future
            robot.schedule_waypoint(
                pose=target_pose,
                target_time=now + 0.05,
            )

            # Progress printout every 50 frames
            if frame_count % 50 == 0:
                cur_state = robot.get_state()
                cur_pos = np.array(cur_state["ActualTCPPose"][:3])
                tgt_pos = target_pose[:3]
                pos_err = np.linalg.norm(cur_pos - tgt_pos) * 1000
                print(f"  [{frame_count:>6}] pos_err={pos_err:.1f}mm")


if __name__ == "__main__":
    main()
