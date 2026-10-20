#!/usr/bin/env python3
"""Five-second subscription-only raw joint temperature/voltage check."""
import json
import time


def main():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from aimdk_msgs.msg import JointStateArray

    rclpy.init(args=[])
    node = rclpy.create_node("x2_joint_thermal_readonly")
    seen = {}
    subscriptions = []

    def receive(message):
        for joint in message.joints:
            values = {field: getattr(joint, field) for field in (
                "position", "velocity", "effort", "coil_temp", "motor_temp", "motor_vol")}
            record = seen.setdefault(joint.name, {"count": 0, "min": dict(values),
                                                   "max": dict(values)})
            record["count"] += 1
            record["last"] = values
            for field, value in values.items():
                record["min"][field] = min(record["min"][field], value)
                record["max"][field] = max(record["max"][field], value)

    for group in ("leg", "waist", "arm", "head"):
        subscriptions.append(node.create_subscription(
            JointStateArray, f"/aima/hal/joint/{group}/state", receive,
            qos_profile_sensor_data))
    started = time.monotonic()
    try:
        while time.monotonic() - started < 5:
            rclpy.spin_once(node, timeout_sec=0.01)
        print(json.dumps({"scope": "subscription-only", "joints": seen}, indent=2))
        return 0 if len(seen) == 31 else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
