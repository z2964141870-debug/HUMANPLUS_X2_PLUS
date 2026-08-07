#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
REPEATS="${REPEATS:-3}"
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-182}"
MOTION=/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz

failures=0
case_index=0
for profile in fixed slow fast; do
  upper_env=()
  if [[ "$profile" == slow ]]; then
    upper_env=(UPPER_MOTION="$MOTION" UPPER_SCALE=0.25 UPPER_TIME_SCALE=0.5 UPPER_MAX_EXCURSION_RAD=0.12 UPPER_MAX_VELOCITY_RADPS=0.20)
  elif [[ "$profile" == fast ]]; then
    upper_env=(UPPER_MOTION="$MOTION" UPPER_SCALE=0.25 UPPER_TIME_SCALE=1.0 UPPER_MAX_EXCURSION_RAD=0.12 UPPER_MAX_VELOCITY_RADPS=0.40)
  fi
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="stage253_left_yawg2p5_${profile}_r${repeat}"
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
      COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 FIXED_WZ=-0.09 MIRROR_POLICY=true \
      ACTION_BIAS_MODE=turn_progress_feedback ACTION_BIAS=1.0 YAW_ACTION_GAIN=2.5 \
      ANKLE_ROLL_COMMON_BIAS=-0.20 STOP_CONTROLLER=policy "${upper_env[@]}" \
      bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
      failures=$((failures + 1))
    fi
  done
done

python3 "$SCRIPT_DIR/summarize_stage253_left_yaw_gain.py" \
  --result-root "$RESULT_ROOT" --repeats "$REPEATS" \
  --output-json "$REPO_ROOT/reports/official_x2/stage253_left_yaw_gain_panel.json" \
  --output-md "$REPO_ROOT/reports/official_x2/stage253_left_yaw_gain_panel.md"

(( failures == 0 ))
