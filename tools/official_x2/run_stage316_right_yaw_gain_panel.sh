#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
MODEL_PATH_VALUE="${MODEL_PATH_VALUE:-/models/stage306_s2652_transition_head_actor.onnx}"
STATIONARY_MODEL_VALUE="${STATIONARY_MODEL_VALUE:-/models/stand_backend_scratch_i150_actor.onnx}"
PREFIX="${PREFIX:-stage316_s2652_split_intent_nominal_right_yaw_gain}"
REPEATS="${REPEATS:-3}"
GAINS="${GAINS:-2.5 3.0 3.5}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/stage316_right_yaw_gain_panel.json}"

failures=0
case_index=0
for gain in $GAINS; do
  gain_tag="${gain//./p}"
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="${PREFIX}_g${gain_tag}_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping existing result $result"
      continue
    fi
    if ! env \
      CASE_NAME="$case_name" \
      ROS_DOMAIN_ID="$((200 + case_index))" \
      RESULT_ROOT="$RESULT_ROOT" \
      MODEL_PATH="$MODEL_PATH_VALUE" \
      STATIONARY_MODEL_PATH="$STATIONARY_MODEL_VALUE" \
      COMMAND_VX=0.30 \
      MOVE_SECONDS=4.0 \
      STOP_SECONDS=8.0 \
      FIXED_WZ=0.15 \
      MIRROR_POLICY=false \
      ACTION_BIAS_MODE=turn_progress_feedback \
      ACTION_BIAS=1.0 \
      YAW_ACTION_GAIN="$gain" \
      ANKLE_ROLL_COMMON_BIAS=0.20 \
      STOP_CONTROLLER=brake_blend_to_policy \
      STOP_TRANSITION_SECONDS=1.0 \
      STOP_INTENT_DECELERATE_SECONDS=2.0 \
      STOP_BRAKE_GAIN=1.5 \
      EVENT_HOLD_MIN_SECONDS=0.5 \
      EVENT_HOLD_SPEED=0.05 \
      PD_PROFILE=official_kp_ankle \
      PD_KP_MULTIPLIER=1.0 \
      PD_KD_MULTIPLIER=1.0 \
      MAX_ATTEMPTS=2 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
      failures=$((failures + 1))
    fi
  done
done

python3 - "$RESULT_ROOT" "$PREFIX" "$REPEATS" "$GAINS" "$failures" "$SUMMARY_PATH" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
repeats = int(sys.argv[3])
gains = [float(value) for value in sys.argv[4].split()]
failures = int(sys.argv[5])
summary_path = pathlib.Path(sys.argv[6])

result = {
    "stage": "stage316_right_yaw_gain_panel",
    "prefix": prefix,
    "attempt_failures": failures,
    "groups": {},
}
for gain in gains:
    gain_tag = str(gain).replace(".", "p")
    rows = []
    for repeat in range(1, repeats + 1):
        path = root / f"{prefix}_g{gain_tag}_r{repeat}.json"
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
            }
        )
    valid = [row for row in rows if row["result"] in ("pass", "fail")]
    result["groups"][str(gain)] = {
        "passes": sum(row["result"] == "pass" for row in valid),
        "valid": len(valid),
        "rows": rows,
    }

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

exit "$failures"
