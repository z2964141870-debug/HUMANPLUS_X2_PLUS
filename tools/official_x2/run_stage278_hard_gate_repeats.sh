#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
MODEL_PATH_VALUE="${MODEL_PATH_VALUE:-/models/stage271_s2620_vxgated_future_actor.onnx}"
PREFIX="${PREFIX:-stage278_s2620_brakeblend10_soft0p9_fast}"
REPEATS="${REPEATS:-3}"
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-220}"

failures=0
for repeat in $(seq 1 "$REPEATS"); do
  case_name="${PREFIX}_r${repeat}"
  if ! env \
    CASE_NAME="$case_name" ROS_DOMAIN_ID="$((BASE_DOMAIN_ID + repeat - 1))" \
    MODEL_PATH="$MODEL_PATH_VALUE" COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 \
    ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.60 \
    ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M=0.08 RECOVERY_EXIT_M=0.03 \
    RECOVERY_SLEW_RATE_PER_S=1.0 STOP_CONTROLLER=brake_blend_to_policy \
    STOP_TRANSITION_SECONDS=1.0 STOP_BRAKE_GAIN=1.5 STOP_BRAKE_LIMIT=0.30 \
    STOP_BRAKE_TEMPLATE_SPEED=0.30 STOP_BRAKE_TEMPLATE_FLOOR=0.25 \
    EVENT_HOLD_MIN_SECONDS=0.5 EVENT_HOLD_SPEED=0.10 \
    PD_KP_MULTIPLIER=0.9 PD_KD_MULTIPLIER=0.9 \
    UPPER_MOTION=/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz \
    UPPER_SCALE=0.25 UPPER_TIME_SCALE=1.0 UPPER_MAX_EXCURSION_RAD=0.12 \
    UPPER_MAX_VELOCITY_RADPS=0.40 \
    bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
    failures=$((failures + 1))
  fi
done

python3 "$SCRIPT_DIR/summarize_stage278_hard_gate.py" \
  --result-root "$RESULT_ROOT" --prefix "$PREFIX" --repeats "$REPEATS" \
  --output-json "$REPO_ROOT/reports/official_x2/stage278_hard_gate_summary.json" \
  --output-md "$REPO_ROOT/reports/official_x2/stage278_hard_gate_summary.md"

(( failures == 0 ))
