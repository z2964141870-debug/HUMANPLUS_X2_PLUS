#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
REPEATS="${REPEATS:-3}"
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-202}"
MOTION=/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz
PREFIX="${PREFIX:-stage257}"
MODEL_PATH_VALUE="${MODEL_PATH_VALUE:-/models/stage219_s2600_actor.onnx}"
ACTION_BIAS_VALUE="${ACTION_BIAS_VALUE:-0.60}"
RECOVERY_ENTER_VALUE="${RECOVERY_ENTER_VALUE:-0.08}"
RECOVERY_EXIT_VALUE="${RECOVERY_EXIT_VALUE:-0.03}"
STOP_CONTROLLER_VALUE="${STOP_CONTROLLER_VALUE:-policy}"
STOP_TRANSITION_SECONDS_VALUE="${STOP_TRANSITION_SECONDS_VALUE:-1.0}"
STOP_BRAKE_GAIN_VALUE="${STOP_BRAKE_GAIN_VALUE:-0.8}"
EVENT_HOLD_SPEED_VALUE="${EVENT_HOLD_SPEED_VALUE:-0.05}"

failures=0
case_index=0
for pd_name in soft0p9 nominal1p0 stiff1p2; do
  case "$pd_name" in
    soft0p9) pd_multiplier=0.9 ;;
    nominal1p0) pd_multiplier=1.0 ;;
    stiff1p2) pd_multiplier=1.2 ;;
  esac
  for upper_name in fixed fast; do
    upper_env=()
    if [[ "$upper_name" == fast ]]; then
      upper_env=(UPPER_MOTION="$MOTION" UPPER_SCALE=0.25 UPPER_TIME_SCALE=1.0 UPPER_MAX_EXCURSION_RAD=0.12 UPPER_MAX_VELOCITY_RADPS=0.40)
    fi
    for repeat in $(seq 1 "$REPEATS"); do
      case_index=$((case_index + 1))
      case_name="${PREFIX}_${pd_name}_${upper_name}_straight_r${repeat}"
      result="$RESULT_ROOT/${case_name}.json"
      if [[ -f "$result" ]] && python3 - "$result" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["summary"].get("full_gate_pass") else 1)
PY
      then
        echo "resume: keeping passed result $result"
        continue
      fi
      if ! env CASE_NAME="$case_name" ROS_DOMAIN_ID="$((BASE_DOMAIN_ID + case_index - 1))" \
        RESULT_ROOT="$RESULT_ROOT" MODEL_PATH="$MODEL_PATH_VALUE" \
        COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 \
        ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS="$ACTION_BIAS_VALUE" \
        ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M="$RECOVERY_ENTER_VALUE" RECOVERY_EXIT_M="$RECOVERY_EXIT_VALUE" \
        RECOVERY_SLEW_RATE_PER_S=1.0 STOP_CONTROLLER="$STOP_CONTROLLER_VALUE" \
        STOP_TRANSITION_SECONDS="$STOP_TRANSITION_SECONDS_VALUE" \
        STOP_BRAKE_GAIN="$STOP_BRAKE_GAIN_VALUE" EVENT_HOLD_SPEED="$EVENT_HOLD_SPEED_VALUE" \
        PD_KP_MULTIPLIER="$pd_multiplier" PD_KD_MULTIPLIER="$pd_multiplier" \
        "${upper_env[@]}" bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
        failures=$((failures + 1))
      fi
    done
  done
done

python3 "$SCRIPT_DIR/summarize_stage257_pd_upper_straight.py" \
  --result-root "$RESULT_ROOT" --repeats "$REPEATS" \
  --prefix "$PREFIX" --action-bias "$ACTION_BIAS_VALUE" \
  --recovery-enter "$RECOVERY_ENTER_VALUE" --recovery-exit "$RECOVERY_EXIT_VALUE" \
  --output-json "$REPO_ROOT/reports/official_x2/${PREFIX}_pd_upper_straight_panel.json" \
  --output-md "$REPO_ROOT/reports/official_x2/${PREFIX}_pd_upper_straight_panel.md"

(( failures == 0 ))
