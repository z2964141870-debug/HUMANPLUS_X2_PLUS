#!/usr/bin/env python3
"""Minimal Stage-1 driver: NO rclpy, sentinel-file only. Max author-like load."""
import argparse, hashlib, subprocess, sys, time
from pathlib import Path

MODEL = Path("/agibot/data/home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1.onnx")
MODEL_SHA = "7dfdec7cf1b1ac67635c3cd8f541c27b50d917df2e3381a6bb086b605e1959a0"

def emit(event, **fields):
    print({"event": event, "mono_s": time.monotonic(), **fields}, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--binary-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    cli = ap.parse_args()
    for p, want in ((cli.binary, cli.binary_sha256), (MODEL, MODEL_SHA)):
        got = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"hash mismatch {p}")
    cli.output.mkdir(parents=True, exist_ok=False)
    argv = [str(cli.binary), "--model", str(MODEL), "--intra-op-threads", "3",
            "--input-type", "zmq", "--zmq-pose-host", "localhost",
            "--zmq-pose-port", "5558", "--zmq-pose-topic", "pose",
            "--zmq-debug-port", "5557", "--zmq-debug-topic", "x2_debug",
            "--autostart-after", "0", "--max-duration", "20",
            "--start-trigger-sentinel", str(cli.output / "start.trigger"),
            "--ready-sentinel", str(cli.output / "ready.sentinel"),
            "--soft-shutdown-trigger-sentinel", str(cli.output / "soft_shutdown.sentinel"),
            "--log-dir", str(cli.output / "csv"), "--flight-recorder-seconds", "60",
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
    emit("spawn")
    log = (cli.output / "runtime.log").open("w")
    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                             cwd="/agibot/data/home/agi/gear-sonic", start_new_session=True)
    deadline = time.monotonic() + 40
    while not (cli.output / "ready.sentinel").exists():
        if child.poll() is not None:
            raise SystemExit("deploy exited before ready")
        if time.monotonic() > deadline:
            raise SystemExit("ready timeout")
        time.sleep(0.2)
    emit("standby_ready")
    marker = cli.output / "mc_frozen.marker"
    deadline = time.monotonic() + 120
    while not marker.exists():
        if child.poll() is not None:
            raise SystemExit("deploy exited while waiting for marker")
        if time.monotonic() > deadline:
            raise SystemExit("marker timeout")
        time.sleep(0.2)
    time.sleep(3.0)  # settle
    (cli.output / "start.trigger").touch(exist_ok=False)
    emit("triggered")
    time.sleep(10.0)  # CONTROL window (no local monitoring load!)
    (cli.output / "soft_shutdown.sentinel").touch(exist_ok=True)
    emit("soft_shutdown_touched")
    try:
        child.wait(timeout=12)
    except subprocess.TimeoutExpired:
        import signal, os
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    emit("child_exit", code=child.returncode)
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
