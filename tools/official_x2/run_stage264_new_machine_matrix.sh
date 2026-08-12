#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/yu/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/stage264_new_machine_freeze_20260812}"
TEMPLATE_PATH="${TEMPLATE_PATH:-/home/yu/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz}"
DOMAIN_BASE="${DOMAIN_BASE:-180}"

python3 "$SCRIPT_DIR/audit_native_locomotion_baseline_preflight.py" >/dev/null
mkdir -p "$RESULT_ROOT"

validate_existing() {
  local path="$1" vx="$2" motion="$3"
  python3 - "$path" "$vx" "$motion" <<'PY'
import json
import math
import sys

path, expected_vx, motion = sys.argv[1], float(sys.argv[2]), sys.argv[3]
summary = json.load(open(path, encoding="utf-8"))["summary"]
assert summary["full_gate_pass"] is True
assert math.isclose(summary["command_vx_mps"], expected_vx, abs_tol=1.0e-12)
if motion == "straight":
    assert summary["move_gate_kind"] == "straight"
    assert summary["fixed_wz_radps"] is None
    assert summary["mirror_policy"] is False
elif motion == "turn_right":
    assert summary["move_gate_kind"] == "turn"
    assert math.isclose(summary["fixed_wz_radps"], 0.15, abs_tol=1.0e-12)
    assert summary["mirror_policy"] is False
else:
    assert summary["move_gate_kind"] == "turn"
    assert math.isclose(summary["fixed_wz_radps"], -0.09, abs_tol=1.0e-12)
    assert summary["mirror_policy"] is True
PY
}

run_case() {
  local case_name="$1" domain="$2" vx="$3" motion="$4"
  local output="$RESULT_ROOT/$case_name.json"
  if [[ -s "$output" ]]; then
    validate_existing "$output" "$vx" "$motion"
    echo "reusing verified pass: $output"
    return
  fi
  [[ ! -e "$output" ]] || { echo "refusing non-regular result: $output" >&2; exit 2; }

  local common=(
    CASE_NAME="$case_name" ROS_DOMAIN_ID="$domain"
    OFFICIAL_ROOT="$OFFICIAL_ROOT" RESULT_ROOT="$RESULT_ROOT"
    TEMPLATE_PATH="$TEMPLATE_PATH"
    MODEL_PATH=/models/stage219_s2600_actor.onnx
    COMMAND_VX="$vx" STATE_QOS_DEPTH=1 STATE_PREDICTION_SECONDS=0
    MOVE_SECONDS=4.0 STOP_SECONDS=8.0 STOP_CONTROLLER=policy
    MAX_ATTEMPTS=1 TIMEOUT_SECONDS=90
  )
  local extra=()
  case "$motion" in
    straight)
      extra=(
        HEADING_GAIN=0.50 HEADING_RATE_LIMIT=0.10
        ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.50
        ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M=0.12
        RECOVERY_EXIT_M=0.04 RECOVERY_SLEW_RATE_PER_S=1.0
      )
      ;;
    turn_right)
      extra=(
        FIXED_WZ=0.15 ACTION_BIAS_MODE=turn_progress_feedback
        ACTION_BIAS=1.0 YAW_ACTION_GAIN=2.0 ANKLE_ROLL_COMMON_BIAS=0.20
      )
      ;;
    turn_left)
      extra=(
        MIRROR_POLICY=true FIXED_WZ=-0.09 ACTION_BIAS_MODE=turn_progress_feedback
        ACTION_BIAS=1.0 YAW_ACTION_GAIN=2.0 ANKLE_ROLL_COMMON_BIAS=-0.20
      )
      ;;
    *) echo "unknown motion: $motion" >&2; exit 2 ;;
  esac
  env "${common[@]}" "${extra[@]}" bash "$SCRIPT_DIR/run_official_gate_case.sh"
}

domain="$DOMAIN_BASE"
for speed_vx in "medium:0.30" "low:0.25"; do
  speed="${speed_vx%%:*}"
  vx="${speed_vx#*:}"
  for motion_repeats in "straight:6" "turn_right:3" "turn_left:3"; do
    motion="${motion_repeats%%:*}"
    repeats="${motion_repeats#*:}"
    for repeat in $(seq 1 "$repeats"); do
      run_case "stage264_${speed}_${motion}_r${repeat}" "$domain" "$vx" "$motion"
      domain=$((domain + 1))
    done
  done
done

python3 "$SCRIPT_DIR/summarize_stage264_new_machine_matrix.py" \
  --result-root "$RESULT_ROOT" \
  --output-json "$REPO_ROOT/stage264_frozen_baseline/new_machine_replay.json" \
  --output-md "$REPO_ROOT/stage264_frozen_baseline/new_machine_replay.md"
