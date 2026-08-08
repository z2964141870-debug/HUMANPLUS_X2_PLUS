#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
PREFIX="${PREFIX:-stage328_s2652_preview0p5_stiff_balance_arm}"
REPEATS="${REPEATS:-3}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/stage328_balance_arm_scale_panel.json}"
MOTION=/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz

failures=0
case_index=0
for scale in 0.10 0.15; do
  scale_tag="${scale/./p}"
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="${PREFIX}_s${scale_tag}_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping existing result $result"
      continue
    fi
    if ! env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((109 + case_index))" \
      RESULT_ROOT="$RESULT_ROOT" \
      MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
      STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx \
      COMMAND_VX=0.30 MOVE_SECONDS=4.0 STOP_SECONDS=8.0 \
      ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.6 \
      ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M=0.08 RECOVERY_EXIT_M=0.03 \
      RECOVERY_SLEW_RATE_PER_S=1.0 FUTURE_STOP_PREVIEW_SECONDS=0.5 \
      STOP_CONTROLLER=brake_blend_to_policy STOP_TRANSITION_SECONDS=1.0 \
      STOP_INTENT_DECELERATE_SECONDS=2.0 STOP_BRAKE_GAIN=1.5 \
      EVENT_HOLD_MIN_SECONDS=0.5 EVENT_HOLD_SPEED=0.05 \
      PD_PROFILE=official_kp_ankle PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2 \
      UPPER_MOTION="$MOTION" UPPER_SCALE="$scale" UPPER_INTENT_SCALE=0.0 \
      UPPER_TIME_SCALE=1.0 UPPER_MAX_EXCURSION_RAD=0.12 UPPER_MAX_VELOCITY_RADPS=0.40 \
      MAX_ATTEMPTS=2 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
      failures=$((failures + 1))
    fi
  done
done

python3 - "$RESULT_ROOT" "$PREFIX" "$REPEATS" "$failures" "$SUMMARY_PATH" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
repeats = int(sys.argv[3])
failures = int(sys.argv[4])
summary_path = pathlib.Path(sys.argv[5])
result = {
    "stage": "stage328_balance_arm_scale_panel",
    "domain": "official_mujoco_stiff1p2",
    "upper_intent_scale": 0.0,
    "future_stop_preview_seconds": 0.5,
    "attempt_failures": failures,
    "groups": {},
}
for scale in (0.10, 0.15):
    tag = f"{scale:.2f}".replace(".", "p")
    rows = []
    for repeat in range(1, repeats + 1):
        path = root / f"{prefix}_s{tag}_r{repeat}.json"
        if not path.exists():
            rows.append({"repeat": repeat, "result": "missing"})
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        rows.append({
            "repeat": repeat,
            "result": "pass" if summary.get("full_gate_pass") else "fail",
            "move": bool(summary.get("move_gate_pass")),
            "stop": bool(summary.get("stop_gate_pass")),
            "heading_max_rad": summary.get("move_heading_max_deviation_rad"),
            "stop_drift_m": summary.get("stop_root_xy_drift_m"),
            "upper_excursion_rad": summary.get("upper_target_excursion_abs_max_rad"),
        })
    valid = [row for row in rows if row["result"] in ("pass", "fail")]
    result["groups"][f"{scale:.2f}"] = {
        "passes": sum(row["result"] == "pass" for row in valid),
        "valid": len(valid),
        "rows": rows,
    }
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY

exit "$failures"
