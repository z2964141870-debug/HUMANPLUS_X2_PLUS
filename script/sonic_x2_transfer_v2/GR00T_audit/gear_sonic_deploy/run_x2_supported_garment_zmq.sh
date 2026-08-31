#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_x2_supported_garment_zmq.sh
#
# Supported (gantry) Sonic run driven by the LIVE garment reference instead of
# StandStillReference. This is the powered garment-teleop entry point:
#
#   garment publisher -> localhost ZMQ v5 :5556 -> Sonic reference/tokenizer
#   -> ONNX policy 50 Hz -> safety gates -> HAL writer 250 Hz -> X2
#
# It reuses the --supported-neutral-damped balance envelope VERBATIM (same
# anchor, deviation limits, tilt/vel trips, damping, LPF, target rate) and adds
# ONLY the live-reference transport + powered safety gates:
#
#   --vla --vla-zmq-host 127.0.0.1 --vla-zmq-port 5556 --vla-zmq-topic pose
#   --pose-ref-stale-s 0.5            (SUPPORTED_POLICY starvation watchdog)
#   --strict-reference-velocity       (v5.1: reject frames without explicit
#                                      joint_vel_mj / an incomplete future
#                                      window; never re-estimate velocity)
#   --zmq-warmup-body-frames 40       (policy entry requires a warmed reference)
#   --zmq-entry-max-age-s 0.5         (reference must be fresh at entry)
#   --zmq-entry-min-fresh-s 0.8       (reference must be CONTINUOUSLY fresh; a
#                                      gap resets it so one recovered frame
#                                      cannot re-arm policy)
#
# Diff vs run_x2_supported_neutral_damped.sh is exactly the ZMQ_ARGS block in
# run_x2_suspended_sonic.sh -- nothing in the balance/gain/writer envelope
# changes. See README_X2_GARMENT_ZMQ.md.
#
# SAFETY: this launcher only STAGES the powered garment path. It does NOT send
# 'policy'. Policy entry is still hand-gated at the operator prompt AND by the
# in-binary ZMQ readiness gate (warmed + fresh + continuous + explicit-vel).
# Do not send 'policy' merely because the pose publisher is up.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_GARMENT_ZMQ_START"

cat >&2 <<'EOF'
BLOCKED: this historical direct live-reference launcher is disabled.

It lacks the isolated :5555 -> safety proxy -> :5556 gate and could admit a
fresh but moving/T-Pose reference. The replacement remains blocked until:
  1. the fixed StandStill parent has its required 3x30s + 1x300s acceptance;
  2. an approved parent manifest verifies every runtime artifact hash; and
  3. powered x2_debug is available while policy is still OFF, allowing the
     proxy to reach STANDSTILL_READY before live-reference arming.

See README_X2_GARMENT_POWERED_INTEGRATION.md. This script starts no process.
EOF
exit 4

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Supported Sonic run driven by the LIVE garment reference over localhost
ZMQ v5 (:5556, topic 'pose'). Reuses the neutral_damped balance envelope
verbatim; adds only the reference transport and powered live-reference
safety gates (strict v5.1 velocity, warmup/fresh/continuous entry gate,
0.5 s starvation watchdog -> bounded 2 s return to captured static hold).

Custom PD stays active between policy attempts; 'policy'/'stop' may be
repeated. Only type 'lifted' after re-suspending the robot to restore MC.

Prerequisites before you send 'policy':
  - garment publisher is producing fresh v5 frames on localhost :5556;
  - the launcher log shows a warmed + continuously-fresh body reference;
  - robot feedback is stable and the supported entry state is acceptable.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-garment-zmq
