#!/usr/bin/env bash
set -uo pipefail

# Frozen official AimDK v1.0 full-episode panel for BASE Phase9.  It evaluates
# the original stand backend, the same-seed f000 training control, and f005.
# Signed pitch is reported from the adapter summary but never enters the gate.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
MODEL_ROOT="${MODEL_ROOT:-$OFFICIAL_ROOT/models}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/phase9_stateful_recovery_paired_gate.json}"
REPEATS=5

ADAPTER_SHA="d80900b2e7e1fac98aa586dc44c9dfb3d8561592e93167d931b8177afed7e8f9"
MOVING_SHA="da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c"
SOURCE_STAND_SHA="edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565"

check_sha() {
  local expected="$1"
  local path="$2"
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA mismatch: $path actual=$actual expected=$expected" >&2
    exit 2
  }
}
check_sha "$ADAPTER_SHA" "$SCRIPT_DIR/stage208_official_mujoco_adapter.py"
check_sha "$MOVING_SHA" "$MODEL_ROOT/stage306_s2652_transition_head_actor.onnx"
check_sha "$SOURCE_STAND_SHA" "$MODEL_ROOT/stand_backend_scratch_i150_actor.onnx"

labels=(source f000 f005)
models=(
  stand_backend_scratch_i150_actor.onnx
  phase9_stateful_recovery_f000_u5_actor.onnx
  phase9_stateful_recovery_f005_u5_actor.onnx
)
for model in "${models[@]}"; do
  [[ -f "$MODEL_ROOT/$model" ]] || { echo "missing evaluation model: $MODEL_ROOT/$model" >&2; exit 2; }
done

case_gate_nonzero_count=0
case_index=0
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  backend="${models[$index]}"
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="phase9_stateful_recovery_${label}_matched_stiff1p2_fixed_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping $result"
      continue
    fi
    if ! env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((90 + case_index))" \
      RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
      MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
      STATIONARY_MODEL_PATH="/models/$backend" RECOVERY_MODEL_PATH="/models/$backend" \
      COMMAND_VX=0.30 POLICY_VX_FLOOR=0 PHASE_OFFSET=0 CLOCK_MODE=step \
      CONTROL_MODE=full DEFAULT_POSE_PROFILE=stage208 \
      PREPARE_SECONDS=0.2 STAND_SECONDS=2.0 STATIONARY_CONTROLLER=policy \
      STATIONARY_WARMUP_SECONDS=0 STATIONARY_BLEND=0.5 \
      MOVE_SECONDS=5.2 MOVE_ACCELERATE_SECONDS=1.0 MOVE_TEMPLATE_MULTIPLIER=1.0 \
      STOP_SECONDS=8.0 STOP_CONTROLLER=curriculum_then_policy \
      STOP_TRANSITION_SECONDS=2.0 STOP_INTENT_DECELERATE_SECONDS=2.0 \
      FUTURE_STOP_PREVIEW_SECONDS=0.5 \
      ACTION_BIAS_MODE=none ACTION_BIAS=0 ANKLE_ROLL_COMMON_BIAS=0 \
      LEFT_HIP_YAW_BIAS=0 RIGHT_HIP_YAW_BIAS=0 YAW_ACTION_GAIN=0 \
      LATERAL_POSITION_GAIN=0 LATERAL_VELOCITY_GAIN=0 HEADING_GAIN=0 \
      CROSS_TRACK_HEADING_GAIN=0 STATE_PREDICTION_SECONDS=0 STATE_QOS_DEPTH=10 \
      ACTION_EMA_ALPHA=1 WAIST_TILT_ACTION_MULTIPLIER=1 \
      PD_PROFILE=official_kp_ankle PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2 \
      MAX_ATTEMPTS=2 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
      # run_official_gate_case intentionally exits 2 for a valid physical
      # full-gate failure.  Count it as an outcome, not infrastructure loss.
      case_gate_nonzero_count=$((case_gate_nonzero_count + 1))
    fi
  done
done

python3 - "$RESULT_ROOT" "$SUMMARY_PATH" "$case_gate_nonzero_count" "${models[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
case_gate_nonzero_count = int(sys.argv[3])
models = dict(zip(("source", "f000", "f005"), sys.argv[4:7]))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_row(label, repeat):
    path = root / f"phase9_stateful_recovery_{label}_matched_stiff1p2_fixed_r{repeat}.json"
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
        "heading_max_rad": summary.get("move_heading_max_deviation_rad"),
        "lateral_displacement_m": summary.get("move_lateral_displacement_m"),
        "stop_drift_m": summary.get("stop_root_xy_drift_m"),
        "stop_settle_time_s": summary.get("stop_settle_time_s"),
        # Reporting only. No pitch key participates in full_gate_pass.
        "signed_pitch_mean_rad": {
            phase: summary.get(f"{phase}_root_pitch_mean_rad")
            for phase in ("stand", "startup", "move", "stop")
        },
    }

groups = {}
for label in ("source", "f000", "f005"):
    rows = [read_row(label, repeat) for repeat in range(1, 6)]
    groups[label] = {
        "model": models[label],
        "model_sha256": sha(pathlib.Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models") / models[label]),
        "rows": rows,
        "passes": sum(row["status"] == "pass" for row in rows),
        "valid": sum(row["status"] in {"pass", "fail"} for row in rows),
    }

source = groups["source"]
control = groups["f000"]
candidate = groups["f005"]
all_valid = all(group["valid"] == 5 for group in groups.values())
not_worse = candidate["passes"] >= max(source["passes"], control["passes"])
toward_five = candidate["passes"] == 5 or candidate["passes"] > max(
    source["passes"], control["passes"]
)
result = {
    "stage": "base_phase9_stateful_recovery_paired_u5",
    "hypothesis": "5% exact stateful recovery RSI improves the frozen official matched-event gate without degrading the source or same-seed 0% training control.",
    "intervention": "same source/seed/5 PPO updates; recovery_fraction 0.00 -> 0.05 only",
    "moving_actor_frozen": "stage306_s2652_transition_head_actor.onnx",
    "official_contract": "prepare/stand/matched start+move/curriculum stop/stationary handoff; stiff-fixed; upper fixed",
    "signed_pitch_reporting_only": True,
    "full_gate_fail_count": sum(
        row["status"] == "fail"
        for group in groups.values()
        for row in group["rows"]
    ),
    "groups": groups,
    "decision": {
        "all_15_episodes_valid": all_valid,
        "candidate_not_worse_than_source_and_control": not_worse,
        "candidate_moves_toward_5_of_5": toward_five,
        "phase9_smoke_gate_pass": all_valid and not_worse and toward_five,
        "updates25_unlocked": False,
        "reason": "Phase9 is a 5-update smoke; any further budget requires an explicit post-panel review.",
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
