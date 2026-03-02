#!/usr/bin/env python3
"""ZMQ replay listener that drives a Realman robot arm.

Subscribes to rapid_driver's ZMQ PUB stream, extracts pose messages,
and sends them as waypoints to a RealmanInterpolationController.

Architecture (matches demo_real_robot_iphone.py):
  - Background thread: drains ZMQ messages, updates latest_pose atomically
  - Main loop: fixed-cadence control at target_hz, sends schedule_waypoint

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
import csv
import json
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from diffusion_policy.common.precise_sleep import precise_wait
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
        "--scale", type=float, default=1.0,
        help="Scale factor for phone displacement (default: 1.0, e.g. 0.5 = half motion)")
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
        "--target_hz", type=float, default=10.0,
        help="Control loop frequency Hz (default: 10)")
    parser.add_argument(
        "--command_latency", "-cl", type=float, default=0.01,
        help="Latency between receiving command and executing on Robot in Sec.")
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

    R_remap = REMAP_PRESETS[args.remap]
    use_remap = not np.allclose(R_remap, np.eye(3))
    R_remap_T = R_remap.T

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

    # ── 4. Shared state between ZMQ thread and main loop ─────────────────
    lock = threading.Lock()
    # latest_pose_T: the most recently received 4x4 transform (or None)
    shared = {
        "latest_pose_T": None,
        "pose_recv_count": 0,
        "topic_counts": defaultdict(int),
        "session_event": None,  # dict when new session starts
        "last_msg_wall": None,
    }

    def zmq_receiver_thread():
        """Drain ZMQ messages and update shared state."""
        while True:
            try:
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

                with lock:
                    shared["topic_counts"][topic] += 1

                # Handle /replay/status control frames
                if topic == "/replay/status":
                    try:
                        event = json.loads(payload_bytes)
                        with lock:
                            shared["session_event"] = event
                    except Exception:
                        pass
                    continue

                # Topic filter
                if args.pose_topic not in topic:
                    continue

                # Parse pose
                try:
                    payload = json.loads(payload_bytes)
                    T_msg = _pose_payload_to_transform(payload)
                except Exception:
                    continue

                with lock:
                    shared["latest_pose_T"] = T_msg
                    shared["pose_recv_count"] += 1
                    shared["last_msg_wall"] = time.time()

            except Exception:
                break

    recv_thread = threading.Thread(target=zmq_receiver_thread, daemon=True)
    recv_thread.start()

    # ── 5. Pose logging ──────────────────────────────────────────────────
    pose_log = []

    def save_pose_log():
        if not pose_log:
            print("  No pose data to save.")
            return
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        fname = os.path.join(log_dir, f"pose_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        header = [
            "wall_time", "frame_recv", "frame_sent",
            "tgt_x", "tgt_y", "tgt_z", "tgt_rx", "tgt_ry", "tgt_rz",
            "act_x", "act_y", "act_z", "act_rx", "act_ry", "act_rz",
        ]
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(pose_log)
        print(f"  Pose log saved: {fname} ({len(pose_log)} rows)")

    # ── 6. State ─────────────────────────────────────────────────────────
    anchor_T = None
    first_pose_T = None
    frame_recv_at_anchor = 0
    sent_count = 0
    target_pose = None  # current target pose (6D), persists between cycles

    def reset_session():
        nonlocal anchor_T, first_pose_T, frame_recv_at_anchor, sent_count, target_pose
        if pose_log:
            save_pose_log()
        anchor_T = None
        first_pose_T = None
        frame_recv_at_anchor = 0
        sent_count = 0
        target_pose = None
        pose_log.clear()

    def cleanup():
        if robot is not None:
            try:
                robot.slow_stop()
            except Exception:
                robot.stop()
        if shm_manager is not None:
            shm_manager.shutdown()

    def signal_handler(sig, frame):
        elapsed = time.time() - t_global_start
        print(f"\n{'='*50}")
        print(f"  Statistics ({elapsed:.1f}s)")
        print(f"{'='*50}")
        with lock:
            recv_count = shared["pose_recv_count"]
            tc = dict(shared["topic_counts"])
        print(f"  Pose frames received: {recv_count}, sent to robot: {sent_count} ({args.target_hz:.0f}Hz)")
        for topic in sorted(tc):
            print(f"    {topic}: {tc[topic]}")
        print(f"{'='*50}")
        save_pose_log()
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    t_global_start = time.time()
    dt = 1.0 / args.target_hz
    print(f"\nListening for pose messages (topic filter: '{args.pose_topic}', "
          f"control loop: {args.target_hz}Hz, scale: {args.scale})...\n")

    # ── 7. Fixed-cadence main loop (matches demo_real_robot_iphone.py) ───
    t_loop_start = time.monotonic()
    iter_idx = 0
    prev_recv_count = 0

    while True:
        t_cycle_end = t_loop_start + (iter_idx + 1) * dt
        t_command_target = t_cycle_end + dt  # target time = one cycle ahead

        # ── 7a. Check for session events ─────────────────────────────
        with lock:
            session_event = shared["session_event"]
            shared["session_event"] = None

        if session_event is not None:
            ev = session_event.get("event")
            if ev == "start":
                sid = session_event.get("session_id", "?")
                secs = session_event.get("total_secs", 0)
                msgs = session_event.get("message_count", 0)
                spd = session_event.get("speed", 1.0)
                print(f"\n  === New session: {sid} ({secs:.1f}s, {msgs} msgs, speed={spd:.1f}x) ===")
                reset_session()
            elif ev == "stop":
                sid = session_event.get("session_id", "?")
                print(f"\n  === Session stopped: {sid} (sent {sent_count} frames) ===")

        # ── 7b. Read latest pose from ZMQ thread ─────────────────────
        with lock:
            latest_T = shared["latest_pose_T"]
            recv_count = shared["pose_recv_count"]
            last_msg_wall = shared["last_msg_wall"]

        # Session gap detection
        if last_msg_wall is not None and anchor_T is not None:
            gap = time.time() - last_msg_wall
            if gap > args.session_gap_secs:
                print(f"\n  Session gap ({gap:.1f}s), resetting anchor.")
                reset_session()
                latest_T = None

        # No new pose data yet
        if latest_T is None:
            precise_wait(t_cycle_end, time_func=time.monotonic)
            iter_idx += 1
            continue

        # ── 7c. Anchor on first pose ─────────────────────────────────
        if first_pose_T is None:
            first_pose_T = latest_T.copy()
            frame_recv_at_anchor = recv_count
            if robot is not None:
                state = robot.get_state()
                anchor_pose = state["ActualTCPPose"]
                anchor_T = pose_to_mat(np.array(anchor_pose))
                target_pose = np.array(anchor_pose, dtype=np.float64)
            else:
                anchor_T = np.eye(4)
                target_pose = mat_to_pose(anchor_T)
            anchor_pose_6d = mat_to_pose(anchor_T)
            first_pose_6d = mat_to_pose(first_pose_T)
            print(f"  Anchor set. First pose recorded.")
            print(f"    anchor_pose = [{', '.join(f'{v:.4f}' for v in anchor_pose_6d)}]")
            print(f"    first_mcap  = [{', '.join(f'{v:.4f}' for v in first_pose_6d)}]")
            precise_wait(t_cycle_end, time_func=time.monotonic)
            iter_idx += 1
            continue

        # ── 7d. Only update target if we got new pose data ───────────
        if recv_count > prev_recv_count:
            prev_recv_count = recv_count

            # World-frame deltas
            dp_world = latest_T[:3, 3] - first_pose_T[:3, 3]
            dR_world = latest_T[:3, :3] @ first_pose_T[:3, :3].T

            # Remap to robot frame
            if use_remap:
                dp_robot = R_remap @ dp_world
                dR_robot = R_remap @ dR_world @ R_remap_T
            else:
                dp_robot = dp_world
                dR_robot = dR_world

            # Compose with anchor
            T_abs = np.eye(4)
            T_abs[:3, 3] = anchor_T[:3, 3] + dp_robot * args.scale
            T_abs[:3, :3] = dR_robot @ anchor_T[:3, :3]
            target_pose = mat_to_pose(T_abs)

        # ── 7e. Wait for sample time, then send command ──────────────
        t_sample = t_cycle_end - args.command_latency
        precise_wait(t_sample, time_func=time.monotonic)

        now = time.time()

        if args.dry_run:
            sent_count += 1
            if sent_count % 10 == 0 or sent_count == 1:
                pos = target_pose[:3]
                print(f"  [recv={recv_count:>4} sent={sent_count:>4}] "
                      f"pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            pose_log.append([
                now, recv_count, sent_count,
                *target_pose.tolist(),
                0, 0, 0, 0, 0, 0,
            ])
        else:
            # Schedule waypoint: target time is one dt into the future
            # (same as demo_real_robot_iphone.py: t_command_target)
            robot.schedule_waypoint(
                pose=target_pose,
                target_time=t_command_target - time.monotonic() + time.time(),
            )
            sent_count += 1

            actual_pose = np.array(robot.get_state()["ActualTCPPose"])

            pose_log.append([
                now, recv_count, sent_count,
                *target_pose.tolist(),
                *actual_pose.tolist(),
            ])

            # Progress printout every 10 sent frames (~1s at 10Hz)
            if sent_count % 10 == 0:
                tgt_pos = target_pose[:3]
                act_pos = actual_pose[:3]
                pos_err = np.linalg.norm(act_pos - tgt_pos) * 1000
                print(f"  [recv={recv_count:>4} sent={sent_count:>4}] pos_err={pos_err:.1f}mm  "
                      f"tgt=[{tgt_pos[0]:.4f},{tgt_pos[1]:.4f},{tgt_pos[2]:.4f}]  "
                      f"act=[{act_pos[0]:.4f},{act_pos[1]:.4f},{act_pos[2]:.4f}]")

        # ── 7f. Wait for cycle end ───────────────────────────────────
        precise_wait(t_cycle_end, time_func=time.monotonic)
        iter_idx += 1


if __name__ == "__main__":
    main()
