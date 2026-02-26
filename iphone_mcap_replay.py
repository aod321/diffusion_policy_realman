#!/usr/bin/env python3
"""Replay iPhone trajectory from MCAP directly to Realman robot."""

import argparse

from iphone_replay import add_replay_arguments, replay_from_transforms
from mcap_pose_loader import (
    DEFAULT_POSE_TOPIC_PATTERN,
    load_pose_trajectory_from_mcap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay trajectory from MCAP pose topic on Realman robot")
    parser.add_argument("mcap", help="Path to input .mcap file")
    parser.add_argument("--pose-topic", default=None,
                        help="Pose topic to read (default: auto select)")
    parser.add_argument("--topic-auto-pattern", default=DEFAULT_POSE_TOPIC_PATTERN,
                        help=f"Regex for auto pose topic candidates (default: {DEFAULT_POSE_TOPIC_PATTERN})")
    parser.add_argument("--timestamp-source", default="payload",
                        choices=["payload", "log_time"],
                        help="Timestamp source: payload timestamp or MCAP log_time")
    add_replay_arguments(parser, include_trajectory_argument=False)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading MCAP: {args.mcap}")
    try:
        timestamps, transforms, topic, info = load_pose_trajectory_from_mcap(
            mcap_path=args.mcap,
            pose_topic=args.pose_topic,
            topic_auto_pattern=args.topic_auto_pattern,
            timestamp_source=args.timestamp_source,
        )
    except Exception as exc:
        parser.error(str(exc))

    print(f"  Pose topic: {topic}")
    candidate_topics = info.get("candidate_topics") or {}
    if candidate_topics:
        print("  Candidate pose topics:")
        for topic_name, count in sorted(candidate_topics.items(), key=lambda x: (-x[1], x[0])):
            print(f"    - {topic_name}: {count} msgs")
    print(f"  Topic messages: {info.get('topic_messages', 0)}")
    print(f"  Parsed valid pose msgs: {info.get('parsed_messages', 0)}")
    print(f"  Dropped invalid msgs: {info.get('dropped_invalid', 0)}")
    print(f"  Removed non-increasing timestamps: {info.get('removed_non_increasing', 0)}")

    replay_from_transforms(
        timestamps=timestamps,
        transforms=transforms,
        args=args,
    )


if __name__ == "__main__":
    main()
