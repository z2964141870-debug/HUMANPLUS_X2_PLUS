#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
PREFIX="${PREFIX:-stage219_closed_loop_gate_v2}"
REPEATS="${REPEATS:-2}"
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-160}"
RESUME="${RESUME:-false}"

MAX_DOMAIN_ID=$((BASE_DOMAIN_ID + REPEATS * 3 - 1))
if (( BASE_DOMAIN_ID < 0 || MAX_DOMAIN_ID > 232 )); then
  echo "ROS domain range ${BASE_DOMAIN_ID}-${MAX_DOMAIN_ID} is invalid; CycloneDDS requires 0-232" >&2
  exit 2
fi

run_case() {
  local skill="$1" repeat="$2" domain="$3"
  shift 3
  local result="$RESULT_ROOT/${PREFIX}_${skill}_r${repeat}.json"
  if [[ "$RESUME" == "true" && -f "$result" ]] && python3 - "$result" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["summary"].get("full_gate_pass") else 1)
PY
  then
    echo "resume: keeping passed result $result"
    return 0
  fi
  env CASE_NAME="${PREFIX}_${skill}_r${repeat}" ROS_DOMAIN_ID="$domain" \
    RESULT_ROOT="$RESULT_ROOT" MODEL_PATH=/models/stage219_s2600_actor.onnx \
    COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 \
    "$@" bash "$SCRIPT_DIR/run_official_gate_case.sh"
}

for repeat in $(seq 1 "$REPEATS"); do
  offset=$(( (repeat - 1) * 3 ))
  if ! run_case straight "$repeat" "$((BASE_DOMAIN_ID + offset))" \
    ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.50 \
    ANKLE_ROLL_COMMON_BIAS=0.20 \
    RECOVERY_ENTER_M=0.12 RECOVERY_EXIT_M=0.04 RECOVERY_SLEW_RATE_PER_S=1.0 \
    STOP_CONTROLLER=policy; then
    echo "straight repeat $repeat failed; continuing matrix" >&2
  fi
  if ! run_case turn_right "$repeat" "$((BASE_DOMAIN_ID + offset + 1))" \
    FIXED_WZ=0.15 ACTION_BIAS_MODE=hip_yaw_common ACTION_BIAS=0.50 \
    ANKLE_ROLL_COMMON_BIAS=0.20 \
    STOP_CONTROLLER=policy; then
    echo "turn_right repeat $repeat failed; continuing matrix" >&2
  fi
  if ! run_case turn_left "$repeat" "$((BASE_DOMAIN_ID + offset + 2))" \
    MIRROR_POLICY=true FIXED_WZ=-0.09 ACTION_BIAS_MODE=hip_yaw_common ACTION_BIAS=-0.50 \
    ANKLE_ROLL_COMMON_BIAS=-0.20 \
    STOP_CONTROLLER=policy; then
    echo "turn_left repeat $repeat failed; continuing matrix" >&2
  fi
done

python3 "$SCRIPT_DIR/summarize_official_gate_matrix.py" \
  --result-root "$RESULT_ROOT" --prefix "$PREFIX" \
  --output-json "$RESULT_ROOT/${PREFIX}_summary.json" \
  --output-md "$RESULT_ROOT/${PREFIX}_summary.md"
