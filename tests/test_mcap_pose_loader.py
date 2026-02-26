import numpy as np
import pytest

from mcap_pose_loader import (
    _cleanup_pose_trajectory,
    _pose_payload_to_transform,
    choose_pose_topic,
)


def test_choose_pose_topic_by_max_count():
    topic = choose_pose_topic({
        "/a/pose": 10,
        "/b/pose": 20,
    })
    assert topic == "/b/pose"


def test_choose_pose_topic_tie_raises():
    with pytest.raises(ValueError):
        choose_pose_topic({
            "/a/pose": 10,
            "/b/pose": 10,
        })


def test_pose_payload_to_transform_identity():
    payload = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }
    T = _pose_payload_to_transform(payload)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_cleanup_pose_trajectory_sort_and_dedup():
    timestamps = np.array([3.0, 1.0, 1.0, 2.0], dtype=np.float64)
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 4, axis=0)
    transforms[0, 0, 3] = 30.0
    transforms[1, 0, 3] = 10.0
    transforms[2, 0, 3] = 11.0
    transforms[3, 0, 3] = 20.0

    ts, tf, stats = _cleanup_pose_trajectory(timestamps, transforms)
    assert np.allclose(ts, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(tf[:, 0, 3], np.array([10.0, 20.0, 30.0]))
    assert stats["removed_non_increasing"] == 1
