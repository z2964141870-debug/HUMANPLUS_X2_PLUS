#!/usr/bin/env python3
"""Stage-1 suspended takeover: author-order standby + SIGSTOP silence.

Order (root-cause driven, 2026-10-21):
  baseline(fresh+official ownership) -> spawn deploy STANDBY (entities+ONNX
  while MC runs) -> wait ready.sentinel -> wait for mc_frozen.marker (an
  externally verified SIGSTOP of the exact MC pid, done by the bounded
  pidfd tool on SoC0) -> prove command silence + feedback freshness ->
  touch start.trigger -> CONTROL window -> touch soft_shutdown sentinel ->
  confirm child exit. NEVER calls the EM stop_app/start_app HTTP API: on
  firmware rc7 that path kills MC and poisons FastDDS SHM (stage0 evidence).
  MC resume is owned by the bounded pause tool's auto-CONT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path("/agibot/data/home/agi/gear-sonic")
MODEL = ROOT / "policies/x2_sonic_s1ft16000_g1.onnx"
MODEL_SHA = "7dfdec7cf1b1ac67635c3cd8f541c27b50d917df2e3381a6bb086b605e1959a0"
CONTROL_WINDOW_S = 4.0
SETTLE_S = 3.0
QUIET_WAIT_S = 35.0  # empirically commands stop 13-24s after SIGSTOP (watchdog in the command publisher)


def emit(event, **fields):
    print(json.dumps({"event": event, "wall_ns": time.time_ns(),
                      "mono_s": time.monotonic(), **fields}), flush=True)


def check_hash(path, expected):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"hash mismatch: {path}: {actual}")


def binary_arguments(binary, directory):
    return [str(binary), "--model", str(MODEL), "--intra-op-threads", "3",
            "--input-type", "zmq",
            "--zmq-pose-host", "localhost", "--zmq-pose-port", "5558",
            "--zmq-pose-topic", "pose",
            "--zmq-debug-port", "5557", "--zmq-debug-topic", "x2_debug",
            "--autostart-after", "0", "--max-duration", "8",
            "--return-seconds", "0", "--hold-for-mc-timeout-s", "0",
            "--mc-mode-poll-s", "0",
            "--start-trigger-sentinel", str(directory / "start.trigger"),
            "--ready-sentinel", str(directory / "ready.sentinel"),
            "--soft-shutdown-trigger-sentinel", str(directory / "soft_shutdown.sentinel"),
            "--log-dir", str(directory / "csv"),
            "--flight-recorder-seconds", "60",
            "--action-clip", "20", "--max-target-dev", "0.30",
            "--max-target-dev-leg", "0.70", "--max-target-dev-waist", "0.45",
            "--max-target-dev-arm", "2.00", "--max-target-dev-head", "0.40",
            "--ramp-seconds", "2", "--tilt-cos", "-0.3",
            "--target-lpf-hz", "8", "--target-lpf-hz-leg", "16",
            "--target-lpf-hz-waist", "0", "--target-lpf-hz-arm", "0",
            "--kp-scale-ankle-pitch", "1", "--kp-scale-ankle-roll", "1",
            "--kp-scale-knee", "1", "--kp-scale-waist-yaw", "1",
            "--kp-scale-waist-pr", "2",
            "--kd-scale-ankle-pitch", "1", "--kd-scale-ankle-roll", "1",
            "--kd-scale-knee", "1", "--kd-scale-hip", "1",
            "--kd-scale-waist-yaw", "1", "--kd-scale-waist-pr", "5.51",
            "--wrist-bypass", "ik", "--head-bypass", "ref",
            "--disable-pose-ref-watchdog"]


def stop_child(child, soft_sentinel=None):
    if child.poll() is not None:
        return
    # Graceful first: the soft-shutdown sentinel asks the binary to ramp
    # down and exit itself. A SIGKILLed participant leaves stale FastDDS
    # SHM ports that poison hal_imu/hal_ethercat publishers (2026-10-21
    # run1-run3 evidence); never skip this unless the sentinel is unknown.
    if soft_sentinel is not None:
        try:
            Path(soft_sentinel).touch(exist_ok=True)
            child.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
    os_kill_group(child, signal.SIGINT)
    try:
        child.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    os_kill_group(child, signal.SIGTERM)
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os_kill_group(child, signal.SIGKILL)
        child.wait(timeout=2)


def os_kill_group(child, sig):
    import os
    os.killpg(child.pid, sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", choices=["FULLY_SUSPENDED_STAGE1"])
    cli = parser.parse_args()
    check_hash(cli.binary, cli.binary_sha256)
    check_hash(MODEL, MODEL_SHA)
    cli.output.mkdir(exist_ok=False)
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
    node = rclpy.create_node("x2_stage1_sigstop_supervisor")
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
        for kind, message_type in (("state", JointStateArray),
                                    ("command", JointCommandArray)):
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

    child, log_file = None, None
    try:
        wait_for(lambda: fresh() and all("command_"+g in seen for g in groups), 12)
        # graph node-name discovery can lag endpoint discovery; retry briefly
        owners = ownership()
        for _ in range(40):
            if all(v == ["mc_ros2_node"] for v in owners.values()):
                break
            spin()
            owners = ownership()
            time.sleep(0.25)
        if any(v != ["mc_ros2_node"] for v in owners.values()):
            raise RuntimeError(f"not sole official ownership: {owners}")
        emit("baseline", ownership=owners, binary_sha256=cli.binary_sha256)

        command = binary_arguments(cli.binary, cli.output)
        emit("spawn", argv=command)
        log_file = (cli.output / "runtime.log").open("x")
        child = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                 cwd=ROOT, start_new_session=True)
        wait_for(lambda: (cli.output / "ready.sentinel").exists()
                 or child.poll() is not None, 40)
        if child.poll() is not None:
            raise RuntimeError("deploy exited before ready")
        if not fresh():
            raise RuntimeError("feedback not fresh in standby")
        emit("standby_ready", ownership=ownership(),
             age_s={k: time.monotonic()-v for k, v in seen.items()})

        marker = cli.output / "mc_frozen.marker"
        wait_for(lambda: marker.exists() or child.poll() is not None, 120)
        if child.poll() is not None:
            raise RuntimeError("deploy exited while waiting for MC freeze")
        marker_seen_s = time.monotonic()
        emit("mc_frozen_seen")

        def quiet_and_fresh():
            now = time.monotonic()
            return (now - marker_seen_s >= SETTLE_S and fresh() and
                    all(now - seen["command_"+g] >= 0.30 for g in groups))

        end = time.monotonic() + QUIET_WAIT_S
        next_note = time.monotonic()
        while time.monotonic() < end:
            spin()
            now = time.monotonic()
            if now >= next_note:
                emit("quiet_progress", age_s={k: now-v for k, v in seen.items()},
                     marker_age=now - marker_seen_s)
                next_note = now + 2.0
            if quiet_and_fresh():
                break
        else:
            raise RuntimeError("command silence not achieved within QUIET_WAIT_S")
        emit("silence_proven", age_s={k: time.monotonic()-v for k, v in seen.items()},
             counts=counts)
        (cli.output / "start.trigger").touch(exist_ok=False)
        emit("triggered", planned_control_s=CONTROL_WINDOW_S)

        started = time.monotonic()
        next_report = started
        soft_done = False
        while child.poll() is None:
            spin()
            now = time.monotonic()
            if now - started >= CONTROL_WINDOW_S and not soft_done:
                emit("control_window_complete", seconds=now - started)
                (cli.output / "soft_shutdown.sentinel").touch(exist_ok=False)
                soft_done = True
            if now - started >= CONTROL_WINDOW_S + 10:
                emit("soft_shutdown_overdue", seconds=now - started)
                stop_child(child, soft_sentinel=cli.output / "soft_shutdown.sentinel")
                break
            if now >= next_report:
                emit("sample", counts=counts,
                     age_s={k: now-v for k, v in seen.items()})
                next_report = now + 0.25
        emit("child_exit", code=child.returncode)
        if child.returncode != 0:
            raise RuntimeError(f"deploy exit {child.returncode}")
        emit("feedback_after_exit", fresh=fresh(), counts=counts)
    finally:
        try:
            if child is not None:
                stop_child(child, soft_sentinel=cli.output / "soft_shutdown.sentinel")
                emit("child_confirmed_exited", code=child.returncode)
        finally:
            try:
                if log_file is not None:
                    log_file.close()
            finally:
                try:
                    node.destroy_node()
                finally:
                    rclpy.shutdown()
    emit("end", success=True)


if __name__ == "__main__":
    main()
