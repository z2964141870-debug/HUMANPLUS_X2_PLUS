#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
MODEL_PATH_VALUE="${MODEL_PATH_VALUE:-/models/stage306_s2652_transition_head_actor.onnx}"
STATIONARY_MODEL_VALUE="${STATIONARY_MODEL_VALUE:-/models/stand_backend_scratch_i150_actor.onnx}"
PREFIX="${PREFIX:-stage317_s2652_split_intent_nominal_asymmetric_turn}"
REPEATS="${REPEATS:-3}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/stage317_asymmetric_turn_panel.json}"
RIGHT_YAW_GAIN="${RIGHT_YAW_GAIN:-3.0}"
LEFT_YAW_GAIN="${LEFT_YAW_GAIN:-2.5}"
FUTURE_STOP_PREVIEW_SECONDS="${FUTURE_STOP_PREVIEW_SECONDS:-0.0}"
MOTION=/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz

failures=0
case_index=0
for upper in fixed fast; do
  upper_env=()
  if [[ "$upper" == fast ]]; then
    upper_env=(
      UPPER_MOTION="$MOTION"
      UPPER_SCALE=0.25
      UPPER_TIME_SCALE=1.0
      UPPER_MAX_EXCURSION_RAD=0.12
      UPPER_MAX_VELOCITY_RADPS=0.40
    )
  fi
  for direction in right left; do
    if [[ "$direction" == right ]]; then
      fixed_wz=0.15
      ankle_bias=0.20
      mirror=false
      yaw_gain="$RIGHT_YAW_GAIN"
    else
      fixed_wz=-0.09
      ankle_bias=-0.20
      mirror=true
      yaw_gain="$LEFT_YAW_GAIN"
    fi
    for repeat in $(seq 1 "$REPEATS"); do
      case_index=$((case_index + 1))
      case_name="${PREFIX}_${upper}_${direction}_r${repeat}"
      result="$RESULT_ROOT/${case_name}.json"
      if [[ -f "$result" ]]; then
        echo "resume: keeping existing result $result"
        continue
      fi
      if ! env \
        CASE_NAME="$case_name" \
        ROS_DOMAIN_ID="$((209 + case_index))" \
        RESULT_ROOT="$RESULT_ROOT" \
        MODEL_PATH="$MODEL_PATH_VALUE" \
        STATIONARY_MODEL_PATH="$STATIONARY_MODEL_VALUE" \
        COMMAND_VX=0.30 \
        MOVE_SECONDS=4.0 \
        STOP_SECONDS=8.0 \
        FIXED_WZ="$fixed_wz" \
        MIRROR_POLICY="$mirror" \
        ACTION_BIAS_MODE=turn_progress_feedback \
        ACTION_BIAS=1.0 \
        YAW_ACTION_GAIN="$yaw_gain" \
        ANKLE_ROLL_COMMON_BIAS="$ankle_bias" \
        STOP_CONTROLLER=brake_blend_to_policy \
        STOP_TRANSITION_SECONDS=1.0 \
        STOP_INTENT_DECELERATE_SECONDS=2.0 \
        FUTURE_STOP_PREVIEW_SECONDS="$FUTURE_STOP_PREVIEW_SECONDS" \
        STOP_BRAKE_GAIN=1.5 \
        EVENT_HOLD_MIN_SECONDS=0.5 \
        EVENT_HOLD_SPEED=0.05 \
        PD_PROFILE=official_kp_ankle \
        PD_KP_MULTIPLIER=1.0 \
        PD_KD_MULTIPLIER=1.0 \
        MAX_ATTEMPTS=2 \
        "${upper_env[@]}" \
        bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
        failures=$((failures + 1))
      fi
    done
  done
done

python3 - "$RESULT_ROOT" "$PREFIX" "$REPEATS" "$failures" "$SUMMARY_PATH" "$RIGHT_YAW_GAIN" "$LEFT_YAW_GAIN" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
repeats = int(sys.argv[3])
failures = int(sys.argv[4])
summary_path = pathlib.Path(sys.argv[5])
right_yaw_gain = float(sys.argv[6])
left_yaw_gain = float(sys.argv[7])

result = {
    "stage": "stage317_asymmetric_turn_panel",
    "prefix": prefix,
    "contract": {"right_yaw_gain": right_yaw_gain, "left_yaw_gain": left_yaw_gain},
    "attempt_failures": failures,
    "groups": {},
}
for upper in ("fixed", "fast"):
    for direction in ("right", "left"):
        rows = []
        for repeat in range(1, repeats + 1):
            path = root / f"{prefix}_{upper}_{direction}_r{repeat}.json"
            if not path.exists():
                rows.append({"repeat": repeat, "result": "missing"})
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
            rows.append(
                {
                    "repeat": repeat,
                    "result": "pass" if summary.get("full_gate_pass") else "fail",
                    "startup": bool(summary.get("startup_gate_pass")),
                    "move": bool(summary.get("move_gate_pass")),
                    "stop": bool(summary.get("stop_gate_pass")),
                    "yaw_ratio": summary.get("turn_yaw_progress_ratio"),
                    "heading_max_deviation_rad": summary.get("move_heading_max_deviation_rad"),
                    "stop_drift_m": summary.get("stop_root_xy_drift_m"),
                    "upper_rmse_rad": summary.get("upper_tracking_rmse_rad"),
                }
            )
        valid = [row for row in rows if row["result"] in ("pass", "fail")]
        result["groups"][f"{upper}_{direction}"] = {
            "passes": sum(row["result"] == "pass" for row in valid),
            "valid": len(valid),
            "rows": rows,
        }

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

exit "$failures"
