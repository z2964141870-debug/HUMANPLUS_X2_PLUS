#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/stage340_recovery_u10_gate.json}"
REPEATS="${REPEATS:-5}"

attempt_failures=0
for repeat in $(seq 1 "$REPEATS"); do
  case_name="stage340_recovery_f010_u10_stiff1p2_fixed_r${repeat}"
  result="$RESULT_ROOT/${case_name}.json"
  if [[ -f "$result" ]]; then
    echo "resume: keeping $result"
    continue
  fi
  if ! env \
    CASE_NAME="$case_name" ROS_DOMAIN_ID="$((190 + repeat))" \
    RESULT_ROOT="$RESULT_ROOT" \
    MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
    STATIONARY_MODEL_PATH=/models/stage339_recovery_f010_u10_actor.onnx \
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

python3 - "$RESULT_ROOT" "$REPEATS" "$attempt_failures" "$SUMMARY_PATH" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
attempt_failures = int(sys.argv[3])
output = pathlib.Path(sys.argv[4])

rows = []
for repeat in range(1, repeats + 1):
    path = root / f"stage340_recovery_f010_u10_stiff1p2_fixed_r{repeat}.json"
    if not path.exists():
        rows.append({"status": "missing", "path": str(path)})
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    rows.append({
        "status": "pass" if summary.get("full_gate_pass") else "fail",
        "path": str(path),
        "stand": bool(summary.get("stand_gate_pass")),
        "startup": bool(summary.get("startup_gate_pass")),
        "move": bool(summary.get("move_gate_pass")),
        "stop": bool(summary.get("stop_gate_pass")),
        "move_heading_max_deviation_rad": summary.get("move_heading_max_deviation_rad"),
        "stop_root_z_min_m": summary.get("stop_root_z_min_m"),
        "stop_tilt_max_rad": summary.get("stop_root_tilt_max_rad"),
        "stop_drift_m": summary.get("stop_root_xy_drift_m"),
        "stop_tail_speed_mps": summary.get("stop_last_1s_mean_speed_mps"),
        "stop_settle_time_s": summary.get("stop_settle_time_s"),
    })

result = {
    "stage": "stage340_recovery_f010_u10_gate",
    "hypothesis": "Five additional matched PPO updates turn the 4/5 Stage337 f010 signal into a strict 5/5 official stiff-fixed gate.",
    "single_variable": "stand/recovery checkpoint Stage337 model_155 -> Stage339 model_160",
    "matched_contract": {
        "moving_model": "stage306_s2652_transition_head_actor.onnx",
        "stationary_model": "stage339_recovery_f010_u10_actor.onnx",
        "pd": "official_kp_ankle x1.2",
        "upper": "fixed",
        "command_vx_mps": 0.30,
    },
    "historical": {"source_stand_backend": "3/5", "stage337_f010_u5": "4/5"},
    "current_invocation_runner_nonzero_count": attempt_failures,
    "rows": rows,
    "passes": sum(row["status"] == "pass" for row in rows),
    "valid": sum(row["status"] in {"pass", "fail"} for row in rows),
}
result["full_gate_failures"] = sum(row["status"] == "fail" for row in rows)
result["promotion_gate_pass"] = result["passes"] == repeats and result["valid"] == repeats
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
