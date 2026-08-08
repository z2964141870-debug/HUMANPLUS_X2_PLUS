#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
PREFIX="${PREFIX:-stage330_zero50_stiff_fixed}"
REPEATS="${REPEATS:-3}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/stage330_zero_upper_checkpoint_screen.json}"
read -r -a STEPS <<< "${STEPS_VALUE:-2654 2657 2660 2662}"
MODEL_PREFIX_VALUE="${MODEL_PREFIX_VALUE:-stage329_s}"
MODEL_SUFFIX_VALUE="${MODEL_SUFFIX_VALUE:-_zero50_transition_actor.onnx}"

failures=0
case_index=0
for step in "${STEPS[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="${PREFIX}_s${step}_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping existing result $result"
      continue
    fi
    if ! env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((119 + case_index))" \
      RESULT_ROOT="$RESULT_ROOT" \
      MODEL_PATH="/models/${MODEL_PREFIX_VALUE}${step}${MODEL_SUFFIX_VALUE}" \
      STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx \
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
      failures=$((failures + 1))
    fi
  done
done

python3 - "$RESULT_ROOT" "$PREFIX" "$REPEATS" "$failures" "$SUMMARY_PATH" "${STEPS[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
repeats = int(sys.argv[3])
failures = int(sys.argv[4])
summary_path = pathlib.Path(sys.argv[5])
steps = [int(value) for value in sys.argv[6:]]
result = {
    "stage": "stage330_zero_upper_checkpoint_screen",
    "hypothesis": "matched zero-upper training improves stiff fixed stop stability",
    "domain": "official_mujoco_stiff1p2",
    "future_stop_preview_seconds": 0.5,
    "attempt_failures": failures,
    "checkpoints": {},
}
for step in steps:
    rows = []
    for repeat in range(1, repeats + 1):
        path = root / f"{prefix}_s{step}_r{repeat}.json"
        if not path.exists():
            rows.append({"repeat": repeat, "result": "missing"})
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        rows.append({
            "repeat": repeat,
            "result": "pass" if summary.get("full_gate_pass") else "fail",
            "startup": bool(summary.get("startup_gate_pass")),
            "move": bool(summary.get("move_gate_pass")),
            "stop": bool(summary.get("stop_gate_pass")),
            "stop_z_min_m": summary.get("stop_root_z_min_m"),
            "stop_tilt_max_rad": summary.get("stop_root_tilt_max_rad"),
            "stop_drift_m": summary.get("stop_root_xy_drift_m"),
        })
    valid = [row for row in rows if row["result"] in ("pass", "fail")]
    result["checkpoints"][str(step)] = {
        "passes": sum(row["result"] == "pass" for row in valid),
        "valid": len(valid),
        "rows": rows,
    }
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY

exit "$failures"
