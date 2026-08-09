#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/stage338_recovery_candidate_panel.json}"
REPEATS="${REPEATS:-3}"

labels=(f000 f005 f010)
models=(
  stage337_recovery_f000_u5_actor.onnx
  stage337_recovery_f005_u5_actor.onnx
  stage337_recovery_f010_u5_actor.onnx
)

attempt_failures=0
case_index=0
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  stationary="${models[$index]}"
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="stage338_recovery_${label}_stiff1p2_fixed_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping $result"
      continue
    fi
    if ! env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((170 + case_index))" \
      RESULT_ROOT="$RESULT_ROOT" \
      MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
      STATIONARY_MODEL_PATH="/models/$stationary" \
      COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 \
      ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.60 \
      ANKLE_ROLL_COMMON_BIAS=0.20 \
      LATERAL_POSITION_GAIN=0.8 LATERAL_VELOCITY_GAIN=0.2 \
      RECOVERY_ENTER_M=0.08 RECOVERY_EXIT_M=0.03 RECOVERY_SLEW_RATE_PER_S=1.0 \
      FUTURE_STOP_PREVIEW_SECONDS=0.5 \
      STOP_CONTROLLER=brake_blend_to_policy STOP_TRANSITION_SECONDS=1.0 \
      STOP_INTENT_DECELERATE_SECONDS=2.0 STOP_BRAKE_GAIN=1.5 \
      EVENT_HOLD_MIN_SECONDS=0.5 EVENT_HOLD_SPEED=0.05 \
      PD_PROFILE=official_kp_ankle PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2 \
      MAX_ATTEMPTS=2 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
      attempt_failures=$((attempt_failures + 1))
    fi
  done
done

python3 - "$RESULT_ROOT" "$REPEATS" "$attempt_failures" "$SUMMARY_PATH" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
attempt_failures = int(sys.argv[3])
output = pathlib.Path(sys.argv[4])
labels = ("f000", "f005", "f010")
result = {
    "stage": "stage338_recovery_candidate_panel",
    "question": "Does low-fraction recovery reset training improve the frozen Stage326 stiff-fixed stop gate?",
    "matched_contract": {
        "moving_model": "stage306_s2652_transition_head_actor.onnx",
        "pd": "official_kp_ankle x1.2",
        "upper": "fixed",
        "command_vx_mps": 0.30,
        "move_seconds": 4.0,
        "stop_seconds": 8.0,
        "future_stop_preview_seconds": 0.5,
        "stop_controller": "brake_blend_to_policy",
    },
    "historical_source_control": {
        "stationary_model": "stand_backend_scratch_i150_actor.onnx",
        "source_glob": str(root / "stage326_s2652_*stiff1p2_fixed_r*.json"),
    },
    "current_invocation_runner_nonzero_count": attempt_failures,
    "candidates": {},
}

def row(path):
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return {
        "status": "pass" if summary.get("full_gate_pass") else "fail",
        "path": str(path),
        "stand": bool(summary.get("stand_gate_pass")),
        "startup": bool(summary.get("startup_gate_pass")),
        "move": bool(summary.get("move_gate_pass")),
        "stop": bool(summary.get("stop_gate_pass")),
        "stop_root_z_min_m": summary.get("stop_root_z_min_m"),
        "stop_tilt_max_rad": summary.get("stop_root_tilt_max_rad"),
        "stop_drift_m": summary.get("stop_root_xy_drift_m"),
        "stop_tail_speed_mps": summary.get("stop_last_1s_mean_speed_mps"),
        "stop_settle_time_s": summary.get("stop_settle_time_s"),
    }

historical = [row(pathlib.Path(path)) for path in sorted(glob.glob(result["historical_source_control"]["source_glob"]))]
result["historical_source_control"]["rows"] = historical
result["historical_source_control"]["passes"] = sum(item["status"] == "pass" for item in historical)
result["historical_source_control"]["valid"] = sum(item["status"] in {"pass", "fail"} for item in historical)
for label in labels:
    rows = [row(root / f"stage338_recovery_{label}_stiff1p2_fixed_r{repeat}.json") for repeat in range(1, repeats + 1)]
    result["candidates"][label] = {
        "rows": rows,
        "passes": sum(item["status"] == "pass" for item in rows),
        "valid": sum(item["status"] in {"pass", "fail"} for item in rows),
    }
result["full_gate_failures"] = sum(
    item["status"] == "fail"
    for candidate in result["candidates"].values()
    for item in candidate["rows"]
)

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
