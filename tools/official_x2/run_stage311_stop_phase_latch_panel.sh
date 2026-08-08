#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
MODEL_PATH_VALUE="${MODEL_PATH_VALUE:-/models/stage306_s2652_transition_head_actor.onnx}"
STATIONARY_MODEL_VALUE="${STATIONARY_MODEL_VALUE:-/models/stand_backend_scratch_i150_actor.onnx}"
REPEATS="${REPEATS:-5}"
EVENT_HOLD_MIN_SECONDS_VALUE="${EVENT_HOLD_MIN_SECONDS_VALUE:-0.90}"
STOP_CONTROLLER_VALUE="${STOP_CONTROLLER_VALUE:-brake_blend_to_policy}"
STOP_TRANSITION_SECONDS_VALUE="${STOP_TRANSITION_SECONDS_VALUE:-1.0}"
STOP_INTENT_DECELERATE_SECONDS_VALUE="${STOP_INTENT_DECELERATE_SECONDS_VALUE:-2.0}"
PD_MULTIPLIER_VALUE="${PD_MULTIPLIER_VALUE:-1.2}"
PD_NAME="${PD_NAME:-stiff1p2}"
PREFIX="${PREFIX:-stage311_s2652_next_ds}"
MOTION=/repo/assets/official_x2/upper_swing_arms_stand_14dof.npz

failures=0
case_index=0
for upper_name in fixed fast; do
  upper_env=()
  if [[ "$upper_name" == fast ]]; then
    upper_env=(
      UPPER_MOTION="$MOTION"
      UPPER_SCALE=0.25
      UPPER_TIME_SCALE=1.0
      UPPER_MAX_EXCURSION_RAD=0.12
      UPPER_MAX_VELOCITY_RADPS=0.40
    )
  fi
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="${PREFIX}_${PD_NAME}_${upper_name}_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping existing result $result"
      continue
    fi
    # CycloneDDS' UDP port mapping overflows above domain 232.  Keep the
    # panel in a valid, dedicated range so startup failures cannot be
    # mistaken for controller failures.
    if ! env \
      CASE_NAME="$case_name" \
      ROS_DOMAIN_ID="$((179 + case_index))" \
      RESULT_ROOT="$RESULT_ROOT" \
      MODEL_PATH="$MODEL_PATH_VALUE" \
      STATIONARY_MODEL_PATH="$STATIONARY_MODEL_VALUE" \
      COMMAND_VX=0.30 \
      MOVE_SECONDS=4.0 \
      STOP_SECONDS=8.0 \
      ACTION_BIAS_MODE=lateral_recovery_supervisor \
      ACTION_BIAS=0.60 \
      ANKLE_ROLL_COMMON_BIAS=0.20 \
      RECOVERY_ENTER_M=0.08 \
      RECOVERY_EXIT_M=0.03 \
      RECOVERY_SLEW_RATE_PER_S=1.0 \
      STOP_CONTROLLER="$STOP_CONTROLLER_VALUE" \
      STOP_TRANSITION_SECONDS="$STOP_TRANSITION_SECONDS_VALUE" \
      STOP_INTENT_DECELERATE_SECONDS="$STOP_INTENT_DECELERATE_SECONDS_VALUE" \
      STOP_BRAKE_GAIN=1.5 \
      STOP_BRAKE_LIMIT=0.30 \
      STOP_BRAKE_TEMPLATE_SPEED=0.30 \
      STOP_BRAKE_TEMPLATE_FLOOR=0.25 \
      EVENT_HOLD_MIN_SECONDS="$EVENT_HOLD_MIN_SECONDS_VALUE" \
      EVENT_HOLD_SPEED=0.05 \
      PD_PROFILE=official_kp_ankle \
      PD_KP_MULTIPLIER="$PD_MULTIPLIER_VALUE" \
      PD_KD_MULTIPLIER="$PD_MULTIPLIER_VALUE" \
      MAX_ATTEMPTS=2 \
      "${upper_env[@]}" \
      bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
      failures=$((failures + 1))
    fi
  done
done

python3 - "$RESULT_ROOT" "$PREFIX" "$REPEATS" "$failures" "$PD_NAME" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
repeats = int(sys.argv[3])
infrastructure_or_gate_failures = int(sys.argv[4])
pd_name = sys.argv[5]
summary = {
    "prefix": prefix,
    "repeats": repeats,
    "infrastructure_or_gate_failures": infrastructure_or_gate_failures,
    "groups": {},
}
for upper in ("fixed", "fast"):
    rows = []
    for repeat in range(1, repeats + 1):
        path = root / f"{prefix}_{pd_name}_{upper}_r{repeat}.json"
        if not path.exists():
            rows.append({"repeat": repeat, "result": "missing"})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))["summary"]
        rows.append(
            {
                "repeat": repeat,
                "result": "pass" if data.get("full_gate_pass") else "fail",
                "startup": bool(data.get("startup_gate_pass")),
                "move": bool(data.get("move_gate_pass")),
                "stop": bool(data.get("stop_gate_pass")),
                "stop_hold_latch_s": data.get("stop_hold_latch_s"),
                "stop_root_z_min_m": data.get("stop_root_z_min_m"),
                "move_heading_max_deviation_rad": data.get("move_heading_max_deviation_rad"),
            }
        )
    summary["groups"][upper] = {
        "passes": sum(row["result"] == "pass" for row in rows),
        "valid": sum(row["result"] in ("pass", "fail") for row in rows),
        "rows": rows,
    }
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

exit "$failures"
