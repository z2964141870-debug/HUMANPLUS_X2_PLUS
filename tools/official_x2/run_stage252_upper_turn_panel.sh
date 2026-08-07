#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
REPEATS="${REPEATS:-3}"
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-162}"
MOTION="/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz"

failures=0
case_index=0
for profile in slow fast; do
  if [[ "$profile" == slow ]]; then
    time_scale=0.5
    max_velocity=0.20
  else
    time_scale=1.0
    max_velocity=0.40
  fi
  for direction in right left; do
    if [[ "$direction" == right ]]; then
      fixed_wz=0.15
      ankle_bias=0.20
      mirror=false
    else
      fixed_wz=-0.09
      ankle_bias=-0.20
      mirror=true
    fi
    for repeat in $(seq 1 "$REPEATS"); do
      case_index=$((case_index + 1))
      case_name="stage252_upper_${profile}_${direction}_r${repeat}"
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
        RESULT_ROOT="$RESULT_ROOT" MODEL_PATH=/models/stage219_s2600_actor.onnx \
        COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 FIXED_WZ="$fixed_wz" \
        MIRROR_POLICY="$mirror" ACTION_BIAS_MODE=turn_progress_feedback ACTION_BIAS=1.0 \
        YAW_ACTION_GAIN=2.0 ANKLE_ROLL_COMMON_BIAS="$ankle_bias" STOP_CONTROLLER=policy \
        UPPER_MOTION="$MOTION" UPPER_SCALE=0.25 UPPER_TIME_SCALE="$time_scale" \
        UPPER_MAX_EXCURSION_RAD=0.12 UPPER_MAX_VELOCITY_RADPS="$max_velocity" \
        bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
        failures=$((failures + 1))
      fi
    done
  done
done

python3 "$SCRIPT_DIR/summarize_stage252_upper_turn.py" \
  --result-root "$RESULT_ROOT" --repeats "$REPEATS" \
  --output-json "$REPO_ROOT/reports/official_x2/stage252_upper_turn_panel.json" \
  --output-md "$REPO_ROOT/reports/official_x2/stage252_upper_turn_panel.md"

(( failures == 0 ))
