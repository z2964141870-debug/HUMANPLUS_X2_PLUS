# Live-Reference Gate Replay

Date: 2026-08-30

This stage is offline only. It started no ROS, ZMQ, MC, or HAL process.

## Input

```text
analysis_logs/garment_live_dryrun_20260830_135137/hmcp_capture.jsonl
frames: 1379
duration: 39.36 s
source parent: 35.02 Hz passing offload dry-run
```

The replay reconstructs the same `pose_scale=0.7` X2 reference and timestamp
velocity used by the passing adapter. Robot yaw `+32.62 deg` is supplied only
to exercise frame math; it is not real feedback and cannot establish powered
safety.

## Result

```text
states seen: WARMUP only
ever STANDSTILL_READY: no
ever LIVE: no
initial rejection: arm offset 1.009 rad > 0.650 rad (calibration T-Pose)
final rejection: waist offset 0.364 rad > 0.200 rad
```

The historical direct launcher could regard this stream as ready because its
entry checks covered frame count/freshness, not wearer posture and stillness.
The isolated proxy rejects the same capture before publishing any guarded
frame. This closes the specific T-Pose/fresh-but-moving entry defect in
offline replay.

It does not prove that the thresholds are a general teleoperation envelope.
The first powered candidate remains stationary-only, and the fixed StandStill
parent plus pre-policy `x2_debug` prerequisite remain open.

## Reproduce

```bash
python3 scripts/garment_zmq/replay_live_reference_gate.py \
  --capture analysis_logs/garment_live_dryrun_20260830_135137/hmcp_capture.jsonl \
  --robot-yaw-deg 32.62 \
  --require-never-ready \
  --output /tmp/garment_gate_replay.json
```

The retained machine-readable result is `replay.json` in this directory.
