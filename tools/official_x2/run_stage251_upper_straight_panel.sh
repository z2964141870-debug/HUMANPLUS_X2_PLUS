#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
REPEATS="${REPEATS:-3}"
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-153}"
MOTION="/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz"

profiles=(slow fast)
failures=0
case_index=0
for profile in "${profiles[@]}"; do
  if [[ "$profile" == slow ]]; then
    time_scale=0.5
    max_velocity=0.20
  else
    time_scale=1.0
    max_velocity=0.40
  fi
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="stage251_upper_${profile}_straight_r${repeat}"
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
      COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 \
      ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.50 \
      ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M=0.12 RECOVERY_EXIT_M=0.04 \
      RECOVERY_SLEW_RATE_PER_S=1.0 STOP_CONTROLLER=policy \
      UPPER_MOTION="$MOTION" UPPER_SCALE=0.25 UPPER_TIME_SCALE="$time_scale" \
      UPPER_MAX_EXCURSION_RAD=0.12 UPPER_MAX_VELOCITY_RADPS="$max_velocity" \
      bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
      failures=$((failures + 1))
    fi
  done
done

python3 "$SCRIPT_DIR/summarize_stage251_upper_disturbance.py" \
  --result-root "$RESULT_ROOT" --repeats "$REPEATS" \
  --output-json "$REPO_ROOT/reports/official_x2/stage251_upper_straight_panel.json" \
  --output-md "$REPO_ROOT/reports/official_x2/stage251_upper_straight_panel.md"

(( failures == 0 ))
