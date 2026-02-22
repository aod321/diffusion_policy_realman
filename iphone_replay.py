#!/usr/bin/env python3
"""Replay recorded iPhone ARKit trajectory on Realman robot.

Usage:
    # Record with tcp_server.py first, then replay:
    python iphone_replay.py recordings/session_xxx/trajectory.npy \
        --robot_ip 192.168.0.204 --speed_scale 0.5 --remap realman

    # Dry run (no robot connection):
    python iphone_replay.py recordings/session_xxx/trajectory.npy \
        --robot_ip dummy --dry_run

Pipeline:
    load .npy → normalize_to_relative → apply_remap → resample → speed_scale
    → connect robot → read anchor EEF → absolute = anchor @ relative → execute
"""

import argparse
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from diffusion_policy.real_world.realman_interpolation_controller import (
    RealmanInterpolationController,
)
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode,
)
from diffusion_policy.common.precise_sleep import precise_wait

# ─── Coordinate remap presets ────────────────────────────────────────────────

REMAP_PRESETS = {
    "identity": np.eye(3),
    "realman": np.array([
        [1,  0,  0],   # Robot X = ARKit X (right)
        [0,  0, -1],   # Robot Y = -ARKit Z (forward)
        [0,  1,  0],   # Robot Z = ARKit Y (up)
    ], dtype=float),
    "arkit_to_ros": np.array([
        [0,  0, -1],   # ROS X = -ARKit Z
        [-1, 0,  0],   # ROS Y = -ARKit X
        [0,  1,  0],   # ROS Z = ARKit Y
    ], dtype=float),
}

# ─── Pose utilities ──────────────────────────────────────────────────────────

def mat_to_pose(mat: np.ndarray) -> np.ndarray:
    """4x4 homogeneous matrix → [x, y, z, rx, ry, rz] (axis-angle)."""
    pos = mat[:3, 3]
    rotvec = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec])


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """[x, y, z, rx, ry, rz] → 4x4 homogeneous matrix."""
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    return mat

# ─── Trajectory processing ───────────────────────────────────────────────────

def load_trajectory(path: str):
    """Load .npy from tcp_server.py: {timestamps, transforms(N,4,4)}."""
    data = np.load(path, allow_pickle=True).item()
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    transforms = np.asarray(data["transforms"], dtype=np.float64)
    assert transforms.ndim == 3 and transforms.shape[1:] == (4, 4)
    return timestamps, transforms


def normalize_to_relative(transforms: np.ndarray) -> np.ndarray:
    """T_rel[i] = T(0)^{-1} @ T(i). First frame becomes identity."""
    T0_inv = np.linalg.inv(transforms[0])
    return np.array([T0_inv @ T for T in transforms])


def apply_remap(transforms: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Conjugate remap: T' = R_hom @ T @ R_hom^{-1}."""
    R_hom = np.eye(4)
    R_hom[:3, :3] = R
    R_hom_inv = np.eye(4)
    R_hom_inv[:3, :3] = R.T
    return np.array([R_hom @ T @ R_hom_inv for T in transforms])


def resample_trajectory(timestamps, transforms, target_hz):
    """Resample with linear position + SLERP rotation interpolation."""
    t0, t1 = timestamps[0], timestamps[-1]
    duration = t1 - t0
    n = max(2, int(duration * target_hz) + 1)
    query_t = np.linspace(t0, t1, n)

    positions = transforms[:, :3, 3]
    rotations = Rotation.from_matrix(transforms[:, :3, :3])
    slerp = Slerp(timestamps, rotations)

    new_pos = np.column_stack([
        np.interp(query_t, timestamps, positions[:, i]) for i in range(3)
    ])
    new_rot = slerp(query_t)

    out = np.zeros((n, 4, 4))
    out[:, :3, :3] = new_rot.as_matrix()
    out[:, :3, 3] = new_pos
    out[:, 3, 3] = 1.0
    return query_t, out


def compute_trajectory_stats(timestamps, transforms):
    """Compute and return trajectory statistics dict."""
    positions = transforms[:, :3, 3]
    duration = timestamps[-1] - timestamps[0]
    pos_range = positions.max(axis=0) - positions.min(axis=0)

    # Frame-to-frame velocities
    dt = np.diff(timestamps)
    dp = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    velocities = dp / np.maximum(dt, 1e-6)
    max_vel = velocities.max()
    mean_vel = velocities.mean()

    return {
        "n_frames": len(timestamps),
        "duration": duration,
        "pos_range": pos_range,
        "max_velocity": max_vel,
        "mean_velocity": mean_vel,
        "start_pos": positions[0],
        "end_pos": positions[-1],
    }


def print_trajectory_info(stats, label="Trajectory"):
    """Pretty-print trajectory statistics."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Frames:       {stats['n_frames']}")
    print(f"  Duration:     {stats['duration']:.2f} s")
    print(f"  Range (m):    X={stats['pos_range'][0]:.4f}  "
          f"Y={stats['pos_range'][1]:.4f}  Z={stats['pos_range'][2]:.4f}")
    print(f"  Max velocity: {stats['max_velocity']:.4f} m/s")
    print(f"  Mean velocity:{stats['mean_velocity']:.4f} m/s")
    print(f"  Start pos:    {stats['start_pos']}")
    print(f"  End pos:      {stats['end_pos']}")
    print(f"{'='*50}\n")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded iPhone trajectory on Realman robot")
    parser.add_argument("trajectory", help="Path to .npy trajectory from tcp_server.py")
    parser.add_argument("--robot_ip", "-ri", required=True, help="Realman robot IP")
    parser.add_argument("--frequency", "-f", type=float, default=10.0,
                        help="Control loop frequency Hz (default: 10)")
    parser.add_argument("--target_hz", type=float, default=None,
                        help="Trajectory resample Hz (default: same as --frequency)")
    parser.add_argument("--speed_scale", "-s", type=float, default=1.0,
                        help="Speed multiplier (<1 = slower/safer)")
    parser.add_argument("--remap", default="realman",
                        choices=list(REMAP_PRESETS.keys()),
                        help="Coordinate remap preset (default: realman)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print trajectory info only, no robot connection")
    parser.add_argument("--max_velocity", type=float, default=0.25,
                        help="Safety speed limit m/s (default: 0.25)")
    parser.add_argument("--follow", action=argparse.BooleanOptionalAction,
                        default=False, help="Realman follow mode")
    parser.add_argument("--lookahead", type=float, default=0.0,
                        help="Command-side smoothing seconds (default: 0)")
    parser.add_argument("--pos_only", action="store_true",
                        help="Replay position only, lock rotation at anchor")

    args = parser.parse_args()
    target_hz = args.target_hz or args.frequency
    np.set_printoptions(precision=4, suppress=True)

    # ── 1. Load ──────────────────────────────────────────────────────────
    print(f"Loading: {args.trajectory}")
    timestamps, transforms = load_trajectory(args.trajectory)
    print(f"  Raw: {len(timestamps)} frames, "
          f"duration={timestamps[-1] - timestamps[0]:.2f}s")

    # ── 2. Normalize to relative ─────────────────────────────────────────
    transforms = normalize_to_relative(transforms)

    # ── 3. pos_only: zero out relative rotations ─────────────────────────
    if args.pos_only:
        for i in range(len(transforms)):
            transforms[i][:3, :3] = np.eye(3)
        print("  pos_only: rotations locked to anchor")

    # ── 4. Coordinate remap ──────────────────────────────────────────────
    R_remap = REMAP_PRESETS[args.remap]
    if not np.allclose(R_remap, np.eye(3)):
        transforms = apply_remap(transforms, R_remap)
        print(f"  Remap: {args.remap}")

    # ── 5. Resample ──────────────────────────────────────────────────────
    # Make timestamps relative (start at 0) before resampling
    timestamps = timestamps - timestamps[0]
    timestamps, transforms = resample_trajectory(timestamps, transforms, target_hz)
    print(f"  Resampled to {len(timestamps)} frames @ {target_hz:.1f} Hz")

    # ── 6. Speed scale (stretch time axis) ───────────────────────────────
    if args.speed_scale != 1.0:
        timestamps = timestamps / args.speed_scale
        print(f"  Speed scale: {args.speed_scale}x → duration={timestamps[-1]:.2f}s")

    # ── 7. Stats & safety check ──────────────────────────────────────────
    stats = compute_trajectory_stats(timestamps, transforms)
    print_trajectory_info(stats, label=f"Final trajectory ({args.remap})")

    if stats["max_velocity"] > args.max_velocity:
        print(f"WARNING: Max velocity {stats['max_velocity']:.4f} m/s "
              f"exceeds limit {args.max_velocity} m/s!")
        resp = input("Continue anyway? [y/N] ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            return

    if args.dry_run:
        print("--dry_run: exiting without robot connection.")
        return

    # ── 8. Connect robot & execute ───────────────────────────────────────
    print(f"Connecting to robot at {args.robot_ip} ...")

    with SharedMemoryManager() as shm_manager:
        with RealmanInterpolationController(
            shm_manager=shm_manager,
            robot_ip=args.robot_ip,
            frequency=100,
            max_pos_speed=args.max_velocity * 1.73,
            max_rot_speed=0.5,
            joints_init=None,
            follow=args.follow,
            lookahead_time=args.lookahead,
        ) as robot:
            time.sleep(0.5)
            print("Robot connected.")

            # Read anchor pose (current EEF)
            state = robot.get_state()
            anchor_pose = state['ActualTCPPose']
            T_anchor = pose_to_mat(anchor_pose)
            print(f"Anchor EEF pose: {anchor_pose}")

            # Compute absolute targets: T_anchor @ T_rel[i]
            absolute_targets = [
                mat_to_pose(T_anchor @ transforms[i])
                for i in range(len(transforms))
            ]

            print(f"\nStarting replay: {len(absolute_targets)} waypoints, "
                  f"{timestamps[-1]:.2f}s")
            print("Press Q to abort.\n")

            with KeystrokeCounter() as key_counter:
                t_start = time.time() + 0.5  # 500ms lead time
                stop = False

                for i, pose in enumerate(absolute_targets):
                    # Check keyboard
                    for key_stroke in key_counter.get_press_events():
                        if key_stroke == KeyCode(char='q'):
                            print("\nQ pressed — stopping.")
                            stop = True
                            break
                    if stop:
                        break

                    t_target = t_start + timestamps[i]
                    now = time.time()

                    # Skip waypoints already in the past
                    if t_target > now:
                        robot.schedule_waypoint(
                            pose=np.array(pose),
                            target_time=t_target,
                        )

                    # Progress printout (~1 Hz)
                    if i % max(1, int(target_hz / args.speed_scale)) == 0 or i == len(absolute_targets) - 1:
                        pct = 100.0 * i / max(1, len(absolute_targets) - 1)
                        elapsed = now - t_start
                        cur_state = robot.get_state()
                        cur_pos = cur_state['ActualTCPPose'][:3]
                        tgt_pos = pose[:3]
                        pos_err = np.linalg.norm(cur_pos - tgt_pos) * 1000
                        print(f"  [{pct:5.1f}%] t={elapsed:.1f}/{timestamps[-1]:.1f}s  "
                              f"pos_err={pos_err:.1f}mm")

                    # Wait until this waypoint's scheduled time
                    precise_wait(t_target, time_func=time.time)

                if not stop:
                    # Wait for the last waypoint to be reached
                    remaining = t_start + timestamps[-1] - time.time() + 0.5
                    if remaining > 0:
                        time.sleep(remaining)
                    print("\nReplay complete.")
                # Robot cleanup handled by context manager


if __name__ == '__main__':
    main()
