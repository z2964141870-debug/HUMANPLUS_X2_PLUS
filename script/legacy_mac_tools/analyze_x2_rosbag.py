#!/usr/bin/env python3
import argparse
import math
import sqlite3
from collections import defaultdict

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


COMMON_TOPIC = "/aima/mc/common/state"
JOINT_TOPICS = (
    "/aima/hal/joint/leg/state",
    "/aima/hal/joint/waist/state",
    "/aima/hal/joint/arm/state",
)
IMU_TOPICS = (
    "/aima/hal/imu/chest/state",
    "/aima/hal/imu/torso/state",
)
TARGET_TOPIC = "/aima/mc/joint/retargeting"


def rms(values):
    return math.sqrt(sum(x * x for x in values) / len(values)) if values else float("nan")


def max_abs(values):
    return max((abs(x) for x in values), default=float("nan"))


def read_topics(conn):
    rows = conn.execute("SELECT id, name, type FROM topics").fetchall()
    return {name: (topic_id, get_message(type_name)) for topic_id, name, type_name in rows}


def read_messages(conn, topic_info, topic_name, start_ns=None, end_ns=None):
    topic_id, msg_cls = topic_info[topic_name]
    query = "SELECT timestamp, data FROM messages WHERE topic_id = ?"
    params = [topic_id]
    if start_ns is not None:
        query += " AND timestamp >= ?"
        params.append(start_ns)
    if end_ns is not None:
        query += " AND timestamp < ?"
        params.append(end_ns)
    query += " ORDER BY timestamp"
    for timestamp, data in conn.execute(query, params):
        yield timestamp, deserialize_message(data, msg_cls)


def find_whole_body_intervals(conn, topic_info):
    samples = [(ts, msg.action_info.action_desc) for ts, msg in read_messages(conn, topic_info, COMMON_TOPIC)]
    transitions = []
    previous = None
    for ts, action in samples:
        if action != previous:
            transitions.append((ts, previous, action))
            previous = action

    intervals = []
    start = None
    for ts, old, new in transitions:
        if new == "WHOLE_BODY_TELEOP":
            start = ts
        elif old == "WHOLE_BODY_TELEOP" and start is not None:
            intervals.append((start, ts))
            start = None
    if start is not None and samples:
        intervals.append((start, samples[-1][0]))
    return transitions, intervals


def joint_window_stats(conn, topic_info, start_ns, end_ns):
    values = defaultdict(lambda: {"velocity": [], "effort": [], "position": []})
    for topic in JOINT_TOPICS:
        for _, msg in read_messages(conn, topic_info, topic, start_ns, end_ns):
            for joint in msg.joints:
                item = values[joint.name]
                item["velocity"].append(float(joint.velocity))
                item["effort"].append(float(joint.effort))
                item["position"].append(float(joint.position))
    result = {}
    for name, item in values.items():
        positions = item["position"]
        result[name] = {
            "count": len(positions),
            "vel_rms": rms(item["velocity"]),
            "vel_peak": max_abs(item["velocity"]),
            "effort_rms": rms(item["effort"]),
            "effort_peak": max_abs(item["effort"]),
            "pos_p2p": max(positions) - min(positions) if positions else float("nan"),
        }
    return result


def imu_window_stats(conn, topic_info, start_ns, end_ns):
    result = {}
    for topic in IMU_TOPICS:
        angular = []
        linear = []
        for _, msg in read_messages(conn, topic_info, topic, start_ns, end_ns):
            w = msg.angular_velocity
            a = msg.linear_acceleration
            angular.append(math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z))
            linear.append(math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z))
        result[topic] = {
            "count": len(angular),
            "angular_rms": rms(angular),
            "angular_peak": max_abs(angular),
            "linear_rms": rms(linear),
            "linear_peak": max_abs(linear),
        }
    return result


def target_window_stats(conn, topic_info, start_ns, end_ns):
    rows = []
    timestamps = []
    for ts, msg in read_messages(conn, topic_info, TARGET_TOPIC, start_ns, end_ns):
        if len(msg.data) == 62:
            timestamps.append(ts)
            rows.append([float(x) for x in msg.data])
    if not rows:
        return {}
    ranges = [max(row[i] for row in rows) - min(row[i] for row in rows) for i in range(62)]
    steps = [0.0] * 62
    for before, after in zip(rows, rows[1:]):
        for i in range(62):
            steps[i] = max(steps[i], abs(after[i] - before[i]))
    intervals = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    return {
        "count": len(rows),
        "hz": 1000.0 / (sum(intervals) / len(intervals)) if intervals else float("nan"),
        "root_lower_waist_max_range": max(ranges[:22]),
        "arm_max_range": max(ranges[22:36]),
        "head_hands_max_range": max(ranges[36:58]),
        "root_lower_waist_max_step": max(steps[:22]),
        "arm_max_step": max(steps[22:36]),
        "head_hands_max_step": max(steps[36:58]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db3")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db3}?mode=ro", uri=True)
    topic_info = read_topics(conn)
    transitions, intervals = find_whole_body_intervals(conn, topic_info)
    print("ACTION TRANSITIONS")
    if transitions:
        origin = transitions[0][0]
        for ts, old, new in transitions:
            print(f"  +{(ts-origin)/1e9:8.3f}s {old} -> {new}")
    if not intervals:
        raise SystemExit("No WHOLE_BODY_TELEOP interval found")

    active_start, active_end = intervals[-1]
    print(f"\nANALYZING LAST WHOLE_BODY INTERVAL: duration={(active_end-active_start)/1e9:.3f}s")
    windows = {
        "pre": (active_start - int(2e9), active_start),
        "active": (active_start, active_end),
        "post": (active_end, active_end + int(2e9)),
    }
    joint_stats = {name: joint_window_stats(conn, topic_info, *window) for name, window in windows.items()}
    imu_stats = {name: imu_window_stats(conn, topic_info, *window) for name, window in windows.items()}

    print("\nWORST JOINT VELOCITY INCREASES (active/pre RMS)")
    ranked = []
    for name, active in joint_stats["active"].items():
        pre = joint_stats["pre"].get(name, {})
        pre_rms = pre.get("vel_rms", float("nan"))
        ratio = active["vel_rms"] / max(pre_rms, 1e-6)
        ranked.append((ratio, name, pre, active, joint_stats["post"].get(name, {})))
    for ratio, name, pre, active, post in sorted(ranked, reverse=True)[:31]:
        print(
            f"  {name:30s} ratio={ratio:7.2f} "
            f"vel_rms {pre.get('vel_rms', float('nan')):.4f}->{active['vel_rms']:.4f} "
            f"peak={active['vel_peak']:.4f} pos_p2p={active['pos_p2p']:.4f} "
            f"effort_rms={active['effort_rms']:.3f} post_vel={post.get('vel_rms', float('nan')):.4f}"
        )

    print("\nIMU")
    for topic in IMU_TOPICS:
        pre = imu_stats["pre"].get(topic, {})
        active = imu_stats["active"].get(topic, {})
        post = imu_stats["post"].get(topic, {})
        print(
            f"  {topic}: angular_rms {pre.get('angular_rms', float('nan')):.4f}"
            f"->{active.get('angular_rms', float('nan')):.4f}"
            f"->{post.get('angular_rms', float('nan')):.4f}, "
            f"active_peak={active.get('angular_peak', float('nan')):.4f}"
        )

    target = target_window_stats(conn, topic_info, active_start, active_end)
    print("\nRECEIVED 62D TARGET DURING ACTIVE")
    for key, value in target.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
