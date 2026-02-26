#!/usr/bin/env python3
"""Analyze target-vs-actual replay trajectory error from replay log."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_replay_log(path: str):
    data = np.load(path, allow_pickle=True)
    required = ["target_times", "target_poses", "actual_times", "actual_poses"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in replay log: {missing}")

    target_times = np.asarray(data["target_times"], dtype=np.float64)
    target_poses = np.asarray(data["target_poses"], dtype=np.float64)
    actual_times = np.asarray(data["actual_times"], dtype=np.float64)
    actual_poses = np.asarray(data["actual_poses"], dtype=np.float64)

    metadata = {}
    if "metadata_json" in data:
        try:
            metadata = json.loads(str(data["metadata_json"]))
        except Exception:
            metadata = {}

    if target_times.ndim != 1:
        raise ValueError(f"target_times should be 1D, got shape={target_times.shape}")
    if actual_times.ndim != 1:
        raise ValueError(f"actual_times should be 1D, got shape={actual_times.shape}")
    if target_poses.ndim != 2 or target_poses.shape[1] != 6:
        raise ValueError(f"target_poses should be (N,6), got shape={target_poses.shape}")
    if actual_poses.ndim != 2 or actual_poses.shape[1] != 6:
        raise ValueError(f"actual_poses should be (M,6), got shape={actual_poses.shape}")
    if len(target_times) != len(target_poses):
        raise ValueError("target_times and target_poses length mismatch")
    if len(actual_times) != len(actual_poses):
        raise ValueError("actual_times and actual_poses length mismatch")
    if len(target_times) < 2:
        raise ValueError("Need at least 2 target points")
    if len(actual_times) < 2:
        raise ValueError("Need at least 2 actual points")

    return {
        "target_times": target_times,
        "target_poses": target_poses,
        "actual_times": actual_times,
        "actual_poses": actual_poses,
        "metadata": metadata,
    }


def interpolate_poses(query_times: np.ndarray, src_times: np.ndarray, src_poses: np.ndarray):
    query_times = np.asarray(query_times, dtype=np.float64)
    pos = src_poses[:, :3]
    out_pos = np.column_stack([
        np.interp(query_times, src_times, pos[:, i]) for i in range(3)
    ])

    rot = Rotation.from_rotvec(src_poses[:, 3:])
    slerp = Slerp(src_times, rot)
    out_rot = slerp(query_times).as_rotvec()
    return np.concatenate([out_pos, out_rot], axis=1)


def search_best_time_shift(
        target_times: np.ndarray,
        target_poses: np.ndarray,
        actual_times: np.ndarray,
        actual_poses: np.ndarray,
        max_shift_ms: float,
        shift_steps: int):
    shifts = np.linspace(-max_shift_ms * 1e-3, max_shift_ms * 1e-3, shift_steps)
    t0 = target_times[0]
    t1 = target_times[-1]

    best = None
    for shift in shifts:
        query_t = actual_times + shift
        mask = (query_t >= t0) & (query_t <= t1)
        if mask.sum() < 5:
            continue
        aligned_target = interpolate_poses(query_t[mask], target_times, target_poses)
        pos_err_m = np.linalg.norm(actual_poses[mask, :3] - aligned_target[:, :3], axis=1)
        score = float(np.mean(pos_err_m))
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "shift_s": float(shift),
                "mask": mask,
                "aligned_target": aligned_target,
            }

    if best is None:
        raise RuntimeError("Failed to find valid overlap during time-shift search")
    return best


def compute_error_series(
        aligned_target: np.ndarray,
        actual: np.ndarray):
    pos_err_m = np.linalg.norm(actual[:, :3] - aligned_target[:, :3], axis=1)
    rot_t = Rotation.from_rotvec(aligned_target[:, 3:])
    rot_a = Rotation.from_rotvec(actual[:, 3:])
    rot_err_deg = np.degrees((rot_t.inv() * rot_a).magnitude())
    return pos_err_m, rot_err_deg


def summarize_errors(pos_err_m: np.ndarray, rot_err_deg: np.ndarray):
    pos_mm = pos_err_m * 1000.0
    summary = {
        "n_samples": int(len(pos_err_m)),
        "pos_mean_mm": float(np.mean(pos_mm)),
        "pos_rmse_mm": float(np.sqrt(np.mean(pos_mm ** 2))),
        "pos_p95_mm": float(np.percentile(pos_mm, 95)),
        "pos_max_mm": float(np.max(pos_mm)),
        "rot_mean_deg": float(np.mean(rot_err_deg)),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_err_deg ** 2))),
        "rot_p95_deg": float(np.percentile(rot_err_deg, 95)),
        "rot_max_deg": float(np.max(rot_err_deg)),
    }
    return summary


def save_error_csv(
        path: str,
        times: np.ndarray,
        target: np.ndarray,
        actual: np.ndarray,
        pos_err_m: np.ndarray,
        rot_err_deg: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "target_x", "target_y", "target_z", "target_rx", "target_ry", "target_rz",
            "actual_x", "actual_y", "actual_z", "actual_rx", "actual_ry", "actual_rz",
            "pos_err_mm", "rot_err_deg",
        ])
        for i in range(len(times)):
            writer.writerow([
                float(times[i]),
                *target[i].tolist(),
                *actual[i].tolist(),
                float(pos_err_m[i] * 1000.0),
                float(rot_err_deg[i]),
            ])
    print(f"Saved CSV: {path}")


def plot_results(
        times: np.ndarray,
        target: np.ndarray,
        actual: np.ndarray,
        pos_err_m: np.ndarray,
        rot_err_deg: np.ndarray,
        title: str):
    fig = plt.figure(figsize=(14, 9))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(target[:, 0], target[:, 1], target[:, 2], label="Target", linewidth=2)
    ax3d.plot(actual[:, 0], actual[:, 1], actual[:, 2], label="Actual", linewidth=2, alpha=0.8)
    ax3d.scatter(target[0, 0], target[0, 1], target[0, 2], c="green", marker="o", s=45)
    ax3d.scatter(actual[0, 0], actual[0, 1], actual[0, 2], c="green", marker="x", s=45)
    ax3d.set_title("3D Trajectory")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend()

    ax_pos = fig.add_subplot(2, 2, 2)
    ax_pos.plot(times, pos_err_m * 1000.0)
    ax_pos.set_title("Position Error")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Error (mm)")
    ax_pos.grid(True, alpha=0.3)

    ax_rot = fig.add_subplot(2, 2, 3)
    ax_rot.plot(times, rot_err_deg, color="tab:orange")
    ax_rot.set_title("Orientation Error")
    ax_rot.set_xlabel("Time (s)")
    ax_rot.set_ylabel("Error (deg)")
    ax_rot.grid(True, alpha=0.3)

    ax_hist = fig.add_subplot(2, 2, 4)
    ax_hist.hist(pos_err_m * 1000.0, bins=40, alpha=0.7, label="Pos (mm)")
    ax_hist2 = ax_hist.twinx()
    ax_hist2.hist(rot_err_deg, bins=40, alpha=0.35, color="tab:orange", label="Rot (deg)")
    ax_hist.set_title("Error Histogram")
    ax_hist.set_xlabel("Error")
    ax_hist.set_ylabel("Count (Pos)")
    ax_hist2.set_ylabel("Count (Rot)")
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def plot_axis_comparison(
        times: np.ndarray,
        target: np.ndarray,
        actual: np.ndarray,
        title: str):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    pos_labels = ["X", "Y", "Z"]
    for i, label in enumerate(pos_labels):
        ax = axes[i, 0]
        ax.plot(times, target[:, i], label="Target", linewidth=2.0)
        ax.plot(times, actual[:, i], label="Actual", linewidth=1.8, linestyle="--")
        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Position {label}")

    rot_labels = ["Rx", "Ry", "Rz"]
    target_rot_deg = np.degrees(target[:, 3:])
    actual_rot_deg = np.degrees(actual[:, 3:])
    for i, label in enumerate(rot_labels):
        ax = axes[i, 1]
        ax.plot(times, target_rot_deg[:, i], label="Target", linewidth=2.0)
        ax.plot(times, actual_rot_deg[:, i], label="Actual", linewidth=1.8, linestyle="--")
        ax.set_ylabel(f"{label} (deg)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Rotation {label}")

    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def derive_axis_figure_path(summary_figure_path: str) -> str:
    root, ext = os.path.splitext(summary_figure_path)
    if ext == "":
        ext = ".png"
    return f"{root}_axes{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze replay error between target and actual trajectories")
    parser.add_argument("log", help="Path to replay log .npz generated by iphone_replay.py")
    parser.add_argument("--max-shift-ms", type=float, default=300.0,
                        help="Max absolute time-shift search range in ms (default: 300)")
    parser.add_argument("--shift-steps", type=int, default=121,
                        help="Number of time-shift samples (default: 121)")
    parser.add_argument("--save-fig", type=str, default=None,
                        help="Optional path to save figure (png/pdf)")
    parser.add_argument("--save-axis-fig", type=str, default=None,
                        help="Optional path to save per-axis comparison figure")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Optional path to save aligned error CSV")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display interactive plot window")
    args = parser.parse_args()

    log = load_replay_log(args.log)
    best = search_best_time_shift(
        target_times=log["target_times"],
        target_poses=log["target_poses"],
        actual_times=log["actual_times"],
        actual_poses=log["actual_poses"],
        max_shift_ms=args.max_shift_ms,
        shift_steps=args.shift_steps,
    )

    mask = best["mask"]
    actual = log["actual_poses"][mask]
    actual_times = log["actual_times"][mask]
    target = best["aligned_target"]
    pos_err_m, rot_err_deg = compute_error_series(target, actual)
    summary = summarize_errors(pos_err_m, rot_err_deg)

    print("\n=== Replay Error Summary ===")
    print(f"  Log file:        {args.log}")
    print(f"  Overlap samples: {summary['n_samples']}")
    print(f"  Best time shift: {best['shift_s'] * 1000.0:+.1f} ms")
    print(f"  Pos mean / p95 / max: {summary['pos_mean_mm']:.2f} / {summary['pos_p95_mm']:.2f} / {summary['pos_max_mm']:.2f} mm")
    print(f"  Pos RMSE:        {summary['pos_rmse_mm']:.2f} mm")
    print(f"  Rot mean / p95 / max: {summary['rot_mean_deg']:.2f} / {summary['rot_p95_deg']:.2f} / {summary['rot_max_deg']:.2f} deg")
    print(f"  Rot RMSE:        {summary['rot_rmse_deg']:.2f} deg")

    if args.save_csv:
        save_error_csv(
            path=args.save_csv,
            times=actual_times,
            target=target,
            actual=actual,
            pos_err_m=pos_err_m,
            rot_err_deg=rot_err_deg,
        )

    title = "Replay Target vs Actual"
    meta = log["metadata"]
    if meta:
        title += f" | remap={meta.get('remap', '?')} follow={meta.get('follow', '?')}"

    fig = plot_results(
        times=actual_times,
        target=target,
        actual=actual,
        pos_err_m=pos_err_m,
        rot_err_deg=rot_err_deg,
        title=title,
    )
    axis_fig = plot_axis_comparison(
        times=actual_times,
        target=target,
        actual=actual,
        title=f"{title} | Per-Axis Comparison",
    )

    if args.save_fig:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_fig)) or ".", exist_ok=True)
        fig.savefig(args.save_fig, dpi=160)
        print(f"Saved figure: {args.save_fig}")

    axis_fig_path = args.save_axis_fig
    if axis_fig_path is None and args.save_fig:
        axis_fig_path = derive_axis_figure_path(args.save_fig)
    if axis_fig_path:
        os.makedirs(os.path.dirname(os.path.abspath(axis_fig_path)) or ".", exist_ok=True)
        axis_fig.savefig(axis_fig_path, dpi=160)
        print(f"Saved axis figure: {axis_fig_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
