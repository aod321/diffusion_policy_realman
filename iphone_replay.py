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
import json
import os
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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

def add_replay_arguments(
        parser: argparse.ArgumentParser,
        include_trajectory_argument: bool = True) -> argparse.ArgumentParser:
    """Attach standard replay CLI arguments and return parser."""
    if include_trajectory_argument:
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
    parser.add_argument("--log_replay", type=str, default=None,
                        help="Save replay target/actual trajectory log to .npz")
    parser.add_argument("--log_actual_hz", type=float, default=20.0,
                        help="Actual pose sampling rate for replay log (default: 20)")
    return parser


def preprocess_trajectory_for_replay(
        timestamps: np.ndarray,
        transforms: np.ndarray,
        args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """Normalize/remap/resample trajectory before replay."""
    target_hz = args.target_hz or args.frequency
    timestamps = np.asarray(timestamps, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)

    if len(timestamps) < 2:
        raise ValueError("Need at least 2 trajectory samples for replay")
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected transforms shape (N,4,4), got {transforms.shape}")

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

    return timestamps, transforms, target_hz, stats


def confirm_velocity_safety(stats: dict, max_velocity: float) -> bool:
    """Return True if safe to continue (possibly after user confirmation)."""
    if stats["max_velocity"] <= max_velocity:
        return True
    print(f"WARNING: Max velocity {stats['max_velocity']:.4f} m/s "
          f"exceeds limit {max_velocity} m/s!")
    resp = input("Continue anyway? [y/N] ").strip().lower()
    return resp == 'y'


def save_replay_log(
        path: str,
        target_times_wall: np.ndarray,
        target_poses: np.ndarray,
        actual_times_wall: np.ndarray,
        actual_poses: np.ndarray,
        metadata: dict) -> None:
    """Persist replay traces for offline error analysis."""
    out_path = os.path.abspath(path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if len(target_times_wall) == 0:
        raise ValueError("target_times_wall is empty")
    t0 = float(target_times_wall[0])
    target_times = np.asarray(target_times_wall, dtype=np.float64)
    actual_times = np.asarray(actual_times_wall, dtype=np.float64)
    target_poses = np.asarray(target_poses, dtype=np.float64)
    actual_poses = np.asarray(actual_poses, dtype=np.float64).reshape(-1, 6)

    np.savez_compressed(
        out_path,
        target_times_wall=target_times,
        target_times=target_times - t0,
        target_poses=target_poses,
        actual_times_wall=actual_times,
        actual_times=actual_times - t0,
        actual_poses=actual_poses,
        metadata_json=np.array(json.dumps(metadata)),
    )
    print(f"Saved replay log: {out_path}")


def execute_replay(
        timestamps: np.ndarray,
        transforms: np.ndarray,
        args: argparse.Namespace,
        target_hz: float) -> None:
    """Execute processed trajectory on Realman robot."""
    from diffusion_policy.real_world.realman_interpolation_controller import (
        RealmanInterpolationController,
    )
    from diffusion_policy.real_world.keystroke_counter import (
        KeystrokeCounter,
        KeyCode,
    )

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
            absolute_targets = np.asarray(absolute_targets, dtype=np.float64)

            print(f"\nStarting replay: {len(absolute_targets)} waypoints, "
                  f"{timestamps[-1]:.2f}s")
            print("Press Q to abort.\n")

            with KeystrokeCounter() as key_counter:
                t_start = time.time() + 0.5  # 500ms lead time
                target_times_wall = t_start + timestamps
                stop = False
                scheduled_count = 0

                log_enabled = args.log_replay is not None
                log_interval = 0.0
                if log_enabled:
                    if args.log_actual_hz <= 0:
                        raise ValueError("--log_actual_hz must be > 0 when --log_replay is enabled")
                    log_interval = 1.0 / args.log_actual_hz
                next_log_time = -np.inf
                actual_times_wall = []
                actual_poses = []

                def maybe_log_actual(now: float, force: bool = False):
                    nonlocal next_log_time
                    if not log_enabled:
                        return None
                    if not force and now < next_log_time:
                        return None
                    cur_state = robot.get_state()
                    cur_pose = np.asarray(cur_state['ActualTCPPose'], dtype=np.float64)
                    actual_times_wall.append(now)
                    actual_poses.append(cur_pose.copy())
                    next_log_time = now + log_interval
                    return cur_pose

                maybe_log_actual(time.time(), force=True)

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
                    sampled_pose = maybe_log_actual(now, force=False)

                    # Skip waypoints already in the past
                    if t_target > now:
                        robot.schedule_waypoint(
                            pose=np.array(pose),
                            target_time=t_target,
                        )
                        scheduled_count += 1

                    # Progress printout (~1 Hz)
                    if i % max(1, int(target_hz / args.speed_scale)) == 0 or i == len(absolute_targets) - 1:
                        pct = 100.0 * i / max(1, len(absolute_targets) - 1)
                        elapsed = now - t_start
                        if sampled_pose is None:
                            cur_state = robot.get_state()
                            cur_pos = cur_state['ActualTCPPose'][:3]
                        else:
                            cur_pos = sampled_pose[:3]
                        tgt_pos = pose[:3]
                        pos_err = np.linalg.norm(cur_pos - tgt_pos) * 1000
                        print(f"  [{pct:5.1f}%] t={elapsed:.1f}/{timestamps[-1]:.1f}s  "
                              f"pos_err={pos_err:.1f}mm")

                    # Wait until this waypoint's scheduled time
                    precise_wait(t_target, time_func=time.time)

                if not stop:
                    # Wait for the last waypoint to be reached
                    t_end = t_start + timestamps[-1] + 0.5
                    if log_enabled:
                        while True:
                            now = time.time()
                            if now >= t_end:
                                break
                            maybe_log_actual(now, force=False)
                            sleep_dt = min(log_interval, max(0.0, t_end - now))
                            time.sleep(max(0.001, sleep_dt))
                    else:
                        remaining = t_end - time.time()
                        if remaining > 0:
                            time.sleep(remaining)
                    print("\nReplay complete.")
                maybe_log_actual(time.time(), force=True)

                if log_enabled:
                    save_replay_log(
                        path=args.log_replay,
                        target_times_wall=target_times_wall,
                        target_poses=absolute_targets,
                        actual_times_wall=np.asarray(actual_times_wall, dtype=np.float64),
                        actual_poses=np.asarray(actual_poses, dtype=np.float64),
                        metadata={
                            "robot_ip": args.robot_ip,
                            "remap": args.remap,
                            "target_hz": float(target_hz),
                            "speed_scale": float(args.speed_scale),
                            "max_velocity": float(args.max_velocity),
                            "follow": bool(args.follow),
                            "lookahead": float(args.lookahead),
                            "scheduled_count": int(scheduled_count),
                            "total_targets": int(len(absolute_targets)),
                            "stopped_by_user": bool(stop),
                        },
                    )
                # Robot cleanup handled by context manager


def replay_from_transforms(
        timestamps: np.ndarray,
        transforms: np.ndarray,
        args: argparse.Namespace) -> dict:
    """Run full replay pipeline from in-memory trajectory arrays."""
    np.set_printoptions(precision=4, suppress=True)
    timestamps, transforms, target_hz, stats = preprocess_trajectory_for_replay(
        timestamps=timestamps,
        transforms=transforms,
        args=args,
    )

    if not confirm_velocity_safety(stats, args.max_velocity):
        print("Aborted.")
        return stats

    if args.dry_run:
        if args.log_replay is not None:
            print("--log_replay is ignored in --dry_run mode (no actual robot trajectory).")
        print("--dry_run: exiting without robot connection.")
        return stats

    execute_replay(
        timestamps=timestamps,
        transforms=transforms,
        args=args,
        target_hz=target_hz,
    )
    return stats


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded iPhone trajectory on Realman robot")
    add_replay_arguments(parser, include_trajectory_argument=True)

    args = parser.parse_args()

    # ── 1. Load ──────────────────────────────────────────────────────────
    print(f"Loading: {args.trajectory}")
    timestamps, transforms = load_trajectory(args.trajectory)
    replay_from_transforms(
        timestamps=timestamps,
        transforms=transforms,
        args=args,
    )


if __name__ == '__main__':
    main()
