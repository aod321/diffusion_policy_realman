#!/usr/bin/env python3
"""Load iPhone pose trajectory from MCAP (foxglove.PoseInFrame)."""

import json
import re
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


POSE_SCHEMA_NAME = "foxglove.PoseInFrame"
DEFAULT_POSE_TOPIC_PATTERN = r".*/pose$"


def _load_make_reader():
    try:
        from mcap.reader import make_reader  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Python dependency 'mcap' is required. Install with: pip install mcap zstandard"
        ) from exc
    return make_reader


def _iter_messages(reader, topic: Optional[str] = None) -> Iterator[Tuple[object, object, object]]:
    if topic is None:
        yield from reader.iter_messages()
        return
    try:
        yield from reader.iter_messages(topics=[topic])
        return
    except TypeError:
        pass

    for schema, channel, message in reader.iter_messages():
        if channel is not None and getattr(channel, "topic", None) == topic:
            yield schema, channel, message


def choose_pose_topic(topic_counts: Dict[str, int]) -> str:
    """Choose a topic by highest message count, require tie-break when equal."""
    if not topic_counts:
        raise ValueError("No pose topic candidates found in MCAP")
    max_count = max(topic_counts.values())
    winners = sorted(topic for topic, count in topic_counts.items() if count == max_count)
    if len(winners) > 1:
        joined = ", ".join(winners)
        raise ValueError(
            f"Multiple pose topics have the same max count ({max_count}): {joined}. "
            "Please specify --pose-topic explicitly."
        )
    return winners[0]


def discover_pose_topics(
        mcap_path: str,
        topic_pattern: str = DEFAULT_POSE_TOPIC_PATTERN) -> Dict[str, int]:
    """Return candidate pose topics and message counts."""
    pattern = re.compile(topic_pattern)
    make_reader = _load_make_reader()
    counts: Dict[str, int] = {}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, _message in _iter_messages(reader):
            if schema is None or channel is None:
                continue
            if getattr(schema, "name", None) != POSE_SCHEMA_NAME:
                continue
            topic = getattr(channel, "topic", "")
            if not pattern.search(topic):
                continue
            counts[topic] = counts.get(topic, 0) + 1

    return counts


def _timestamp_from_payload_seconds(payload: dict) -> Optional[float]:
    ts = payload.get("timestamp")
    if not isinstance(ts, dict):
        return None
    sec = ts.get("sec")
    nsec = ts.get("nsec")
    if sec is None or nsec is None:
        return None
    return float(sec) + float(nsec) * 1e-9


def _pose_payload_to_transform(payload: dict) -> np.ndarray:
    pose = payload["pose"]
    position = pose["position"]
    orientation = pose["orientation"]

    xyz = np.array([
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
    ], dtype=np.float64)
    quat_xyzw = np.array([
        float(orientation["x"]),
        float(orientation["y"]),
        float(orientation["z"]),
        float(orientation["w"]),
    ], dtype=np.float64)

    if not np.all(np.isfinite(xyz)):
        raise ValueError("Position contains non-finite values")
    if not np.all(np.isfinite(quat_xyzw)):
        raise ValueError("Quaternion contains non-finite values")

    quat_norm = np.linalg.norm(quat_xyzw)
    if quat_norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    quat_xyzw /= quat_norm

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz
    transform[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    return transform


def _cleanup_pose_trajectory(
        timestamps: np.ndarray,
        transforms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    order = np.argsort(timestamps, kind="mergesort")
    timestamps = timestamps[order]
    transforms = transforms[order]

    removed_non_increasing = 0
    if len(timestamps) >= 2:
        keep = np.ones(len(timestamps), dtype=bool)
        keep[1:] = np.diff(timestamps) > 0
        removed_non_increasing = int((~keep).sum())
        timestamps = timestamps[keep]
        transforms = transforms[keep]

    if len(timestamps) < 2:
        raise ValueError("Need at least 2 valid pose samples after cleanup")

    return timestamps, transforms, {
        "removed_non_increasing": removed_non_increasing,
    }


def load_pose_trajectory_from_mcap(
        mcap_path: str,
        pose_topic: Optional[str] = None,
        topic_auto_pattern: str = DEFAULT_POSE_TOPIC_PATTERN,
        timestamp_source: str = "payload") -> Tuple[np.ndarray, np.ndarray, str, Dict[str, object]]:
    """Load timestamps + 4x4 transforms from MCAP pose topic."""
    if timestamp_source not in {"payload", "log_time"}:
        raise ValueError("--timestamp-source must be one of: payload, log_time")

    candidate_topics: Dict[str, int] = {}
    selected_topic = pose_topic
    if selected_topic is None:
        candidate_topics = discover_pose_topics(
            mcap_path=mcap_path,
            topic_pattern=topic_auto_pattern,
        )
        selected_topic = choose_pose_topic(candidate_topics)

    make_reader = _load_make_reader()
    timestamps = []
    transforms = []
    parsed_messages = 0
    dropped_invalid = 0
    topic_messages = 0

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in _iter_messages(reader, topic=selected_topic):
            if channel is None or getattr(channel, "topic", None) != selected_topic:
                continue
            topic_messages += 1

            if schema is None or getattr(schema, "name", None) != POSE_SCHEMA_NAME:
                dropped_invalid += 1
                continue

            try:
                payload = json.loads(bytes(message.data).decode("utf-8"))
                transform = _pose_payload_to_transform(payload)
                if timestamp_source == "payload":
                    timestamp = _timestamp_from_payload_seconds(payload)
                    if timestamp is None:
                        timestamp = float(message.log_time) * 1e-9
                else:
                    timestamp = float(message.log_time) * 1e-9
            except Exception:
                dropped_invalid += 1
                continue

            timestamps.append(timestamp)
            transforms.append(transform)
            parsed_messages += 1

    if parsed_messages == 0:
        suggestions = discover_pose_topics(
            mcap_path=mcap_path,
            topic_pattern=topic_auto_pattern,
        )
        suggestion_text = ", ".join(sorted(suggestions)) if suggestions else "<none>"
        raise ValueError(
            f"No valid pose samples found for topic '{selected_topic}'. "
            f"Detected pose topics: {suggestion_text}"
        )

    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    transforms_np = np.asarray(transforms, dtype=np.float64)
    timestamps_np, transforms_np, cleanup_stats = _cleanup_pose_trajectory(
        timestamps=timestamps_np,
        transforms=transforms_np,
    )

    info: Dict[str, object] = {
        "candidate_topics": candidate_topics,
        "selected_topic": selected_topic,
        "topic_messages": topic_messages,
        "parsed_messages": parsed_messages,
        "dropped_invalid": dropped_invalid,
    }
    info.update(cleanup_stats)
    return timestamps_np, transforms_np, selected_topic, info
