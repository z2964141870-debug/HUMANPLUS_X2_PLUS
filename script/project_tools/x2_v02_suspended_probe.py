#!/usr/bin/env python3
"""Bounded SoC1-only s1ft16000 probe; powered mode requires full suspension."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path("/agibot/data/home/agi/gear-sonic")
MODEL = ROOT / "policies/x2_sonic_s1ft16000_g1.onnx"
MODEL_SHA = "7dfdec7cf1b1ac67635c3cd8f541c27b50d917df2e3381a6bb086b605e1959a0"
MC_DISCOVERY_SETTLE_S = 3.0
POWERED_WINDOW_S = 3.0


def arguments(binary, directory, powered):
    args = [str(binary), "--model", str(MODEL), "--input-type", "motion_file",
            "--autostart-after", "0", "--max-duration", "3" if powered else "10",
            "--return-seconds", "0", "--hold-for-mc-timeout-s", "0",
            "--mc-mode-poll-s", "0", "--log-dir", str(directory / "csv"),
            "--action-clip", "20", "--max-target-dev", "0.30",
            "--max-target-dev-leg", "0.70", "--max-target-dev-waist", "0.45",
            "--max-target-dev-arm", "2.00", "--max-target-dev-head", "0.50",
            "--ramp-seconds", "2", "--tilt-cos", "-0.3", "--target-lpf-hz", "8",
            "--target-lpf-hz-leg", "16", "--target-lpf-hz-waist", "0",
            "--target-lpf-hz-arm", "0", "--kp-scale-ankle-pitch", "1",
            "--kp-scale-ankle-roll", "1", "--kp-scale-knee", "1",
            "--kp-scale-waist-yaw", "1", "--kp-scale-waist-pr", "2",
            "--kd-scale-ankle-pitch", "1", "--kd-scale-ankle-roll", "1",
            "--kd-scale-knee", "1", "--kd-scale-hip", "1",
            "--kd-scale-waist-yaw", "1", "--kd-scale-waist-pr", "5.51"]
    if powered:
        args += ["--start-trigger-sentinel", str(directory / "start.trigger"),
                 "--ready-sentinel", str(directory / "ready.sentinel")]
    else:
        args += ["--no-publish"]
    return args


def emit(event, **fields):
    print(json.dumps({"event": event, "wall_ns": time.time_ns(),
                      "mono_s": time.monotonic(), **fields}), flush=True)


def check_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"hash mismatch: {path}: {actual}")


def em(action):
    url = "http://10.0.1.40:50080/json/" + action
    request = urllib.request.Request(url, data=b'{"app_name":"mc"}',
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read(65536)
    emit("em_reply", action=action, body=body.decode(errors="replace"))


def stop_child(child):
    if child.poll() is not None:
        return
    # Exact session created by this probe, never a name-based process sweep.
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", choices=["FULLY_SUSPENDED_3S"])
    cli = parser.parse_args()
    powered = bool(cli.execute)
    check_hash(cli.binary, cli.binary_sha256)
    check_hash(MODEL, MODEL_SHA)
    cli.output.mkdir(exist_ok=False)
    # Reject any prior deploy, regardless of its filename or installation path.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv0 = (proc / "cmdline").read_bytes().split(b"\0", 1)[0]
        except (OSError, PermissionError):
            continue
        if argv0.rsplit(b"/", 1)[-1] == b"x2_deploy_onnx_ref":
            raise RuntimeError(f"existing deploy process: {proc.name}")

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Imu
    from aimdk_msgs.msg import JointStateArray, JointCommandArray

    cancelled = False

    def request_cancel(signum, frame):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_cancel)
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("x2_suspended_probe_supervisor")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    seen, counts, subscriptions = {}, {}, []
    groups = ("leg", "waist", "arm", "head")

    def callback(key):
        def receive(message):
            seen[key] = time.monotonic()
            counts[key] = counts.get(key, 0) + 1
        return receive

    for group in groups:
        for kind, message_type in (("state", JointStateArray), ("command", JointCommandArray)):
            subscriptions.append(node.create_subscription(
                message_type, f"/aima/hal/joint/{group}/{kind}",
                callback(f"{kind}_{group}"), qos))
    subscriptions.append(node.create_subscription(
        Imu, "/aima/hal/imu/torso/state", callback("imu"), qos))
    required = ["state_" + g for g in groups] + ["imu"]

    def spin(cancellable=True):
        rclpy.spin_once(node, timeout_sec=0.002)
        if cancellable and cancelled:
            raise RuntimeError("operator/process cancellation")

    def fresh():
        now = time.monotonic()
        return all(key in seen and now - seen[key] <= 0.5 for key in required)

    def wait_for(predicate, seconds, cancellable=True):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            spin(cancellable)
            if predicate():
                return
        raise RuntimeError("precondition timeout")

    def ownership():
        return {g: [e.node_name for e in node.get_publishers_info_by_topic(
            f"/aima/hal/joint/{g}/command")] for g in groups}

    child, restore_needed = None, False
    log_file = None
    try:
        wait_for(lambda: fresh() and all("command_"+g in seen for g in groups), 12)
        owners = ownership()
        if any(v != ["mc_ros2_node"] for v in owners.values()):
            raise RuntimeError(f"not sole official ownership: {owners}")
        emit("baseline", ownership=owners, powered=powered, binary_sha256=cli.binary_sha256)
        command = ["timeout", "-s", "TERM", "-k", "1", "30"] + arguments(
            cli.binary, cli.output, powered)
        emit("spawn", argv=command)
        log_file = (cli.output / "runtime.log").open("x")
        child = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                 cwd=ROOT, start_new_session=True)
        if powered:
            wait_for(lambda: (cli.output / "ready.sentinel").exists() or child.poll() is not None, 15)
            if child.poll() is not None:
                raise RuntimeError("deploy exited before ready")
            if not fresh():
                raise RuntimeError("feedback not fresh before MC stop")
            # A lost HTTP reply may still have stopped MC: always recover after attempting it.
            restore_needed = True
            em("stop_app")
            stop_reply = time.monotonic()

            def quiet_and_fresh():
                now = time.monotonic()
                return (now - stop_reply >= MC_DISCOVERY_SETTLE_S and fresh() and
                        all(now - seen["command_"+g] >= 0.30 for g in groups))

            emit("discovery_settle", seconds=MC_DISCOVERY_SETTLE_S)
            wait_for(quiet_and_fresh, MC_DISCOVERY_SETTLE_S + 3)
            emit("silence_proven", age_s={k: time.monotonic()-v for k, v in seen.items()},
                 counts=counts)
            (cli.output / "start.trigger").touch(exist_ok=False)
            emit("triggered", planned_control_s=3)

        started = time.monotonic()
        next_report = started
        deadline = started + (POWERED_WINDOW_S if powered else 35)
        while child.poll() is None:
            spin()
            now = time.monotonic()
            if now >= deadline:
                if powered:
                    # SAFE_HOLD bypasses native max-duration; bound this suspended probe too.
                    emit("powered_window_complete", seconds=POWERED_WINDOW_S)
                    stop_child(child)
                    break
                raise RuntimeError("bounded deploy deadline exceeded")
            if now >= next_report:
                emit("sample", counts=counts,
                     age_s={k: now-v for k, v in seen.items()})
                next_report = now + 0.25
        emit("child_exit", code=child.returncode)
        if child.returncode != 0:
            raise RuntimeError(f"deploy exit {child.returncode}")
    finally:
        try:
            if child is not None:
                stop_child(child)
                emit("child_confirmed_exited", code=child.returncode)
            if restore_needed:
                # Never restart MC until the exact child session is confirmed gone.
                em("start_app")
                wait_for(lambda: fresh() and all(
                    time.monotonic()-seen.get("command_"+g, 0) < 0.5 for g in groups), 12,
                    cancellable=False)
                wait_for(lambda: all(v == ["mc_ros2_node"] for v in ownership().values()), 45,
                         cancellable=False)
                emit("official_restored", ownership=ownership())
        finally:
            # Release DDS resources even when official recovery cannot be verified.
            try:
                if log_file is not None:
                    log_file.close()
            finally:
                try:
                    node.destroy_node()
                finally:
                    rclpy.shutdown()
    emit("end", success=True, powered=powered)


if __name__ == "__main__":
    main()
