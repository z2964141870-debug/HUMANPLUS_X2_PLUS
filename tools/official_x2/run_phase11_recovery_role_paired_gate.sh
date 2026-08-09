#!/usr/bin/env bash
set -uo pipefail

# Paired official gate for the repaired curriculum-stop role contract.
# Stationary is the frozen source model in both groups. The sole experimental
# variable is the post-transition recovery ONNX.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
MODEL_ROOT="${MODEL_ROOT:-$OFFICIAL_ROOT/models}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/phase11_recovery_role_paired_gate.json}"
REPEATS=5

ADAPTER_SHA="e682da8ca7c18d1a55ac1a65bc9974ed490c24a12704df83ad7e302ac1c5dae3"
HANDOFF_SHA="9277f405b74c455ac525510001f2e89860d9b525f89fd51cffe0aaba62cf4644"
MOVING_SHA="da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c"
SOURCE_STAND_SHA="edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565"
CANDIDATE_SHA="9bc672fc3c535dbe6cd2709cdec4531eb9b61990457172e9c813a8ac713fc0ca"

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
check_sha "$HANDOFF_SHA" "$SCRIPT_DIR/skill_handoff_contract.py"
check_sha "$MOVING_SHA" "$MODEL_ROOT/stage306_s2652_transition_head_actor.onnx"
check_sha "$SOURCE_STAND_SHA" "$MODEL_ROOT/stand_backend_scratch_i150_actor.onnx"
check_sha "$CANDIDATE_SHA" "$MODEL_ROOT/phase9_stateful_recovery_f005_u5_actor.onnx"

labels=(source_recovery candidate_recovery)
recovery_models=(
  stand_backend_scratch_i150_actor.onnx
  phase9_stateful_recovery_f005_u5_actor.onnx
)
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  recovery="${recovery_models[$index]}"
  for repeat in $(seq 1 "$REPEATS"); do
    case_name="phase11_${label}_matched_stiff1p2_fixed_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping $result"
      continue
    fi
    env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((130 + index * 10 + repeat))" \
      RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
      MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
      STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx \
      RECOVERY_MODEL_PATH="/models/$recovery" \
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
      bash "$SCRIPT_DIR/run_official_gate_case.sh" || true
  done
done

python3 - "$RESULT_ROOT" "$SUMMARY_PATH" "${recovery_models[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
models = dict(zip(("source_recovery", "candidate_recovery"), sys.argv[3:5]))
model_root = pathlib.Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def row(label, repeat):
    path = root / f"phase11_{label}_matched_stiff1p2_fixed_r{repeat}.json"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    s = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return {
        "status": "pass" if s.get("full_gate_pass") else "fail",
        "path": str(path),
        "stand": bool(s.get("stand_gate_pass")),
        "startup": bool(s.get("startup_gate_pass")),
        "move": bool(s.get("move_gate_pass")),
        "stop": bool(s.get("stop_gate_pass")),
        "heading_max_rad": s.get("move_heading_max_deviation_rad"),
        "lateral_displacement_m": s.get("move_lateral_displacement_m"),
        "stop_drift_m": s.get("stop_root_xy_drift_m"),
        "stop_settle_time_s": s.get("stop_settle_time_s"),
        "signed_pitch_mean_rad": {
            phase: s.get(f"{phase}_root_pitch_mean_rad")
            for phase in ("stand", "startup", "move", "stop")
        },
        "policy_slot_counts": s.get("policy_slot_inference_counts"),
    }

groups = {}
for label in ("source_recovery", "candidate_recovery"):
    rows = [row(label, repeat) for repeat in range(1, 6)]
    groups[label] = {
        "stationary_model": "stand_backend_scratch_i150_actor.onnx",
        "recovery_model": models[label],
        "recovery_model_sha256": sha(model_root / models[label]),
        "rows": rows,
        "passes": sum(item["status"] == "pass" for item in rows),
        "valid": sum(item["status"] in {"pass", "fail"} for item in rows),
    }

expected_counts = {"main": 360, "stationary": 100, "recovery": 300}
role_contract_pass = all(
    item.get("policy_slot_counts") == expected_counts
    for group in groups.values()
    for item in group["rows"]
    if item["status"] in {"pass", "fail"}
)
source = groups["source_recovery"]
candidate = groups["candidate_recovery"]
result = {
    "stage": "base_phase11_recovery_role_paired_gate",
    "hypothesis": "A source stationary actor protects stand/start while f005 owns only post-transition recovery.",
    "single_variable": "recovery_model source -> phase9 f005",
    "stationary_model_frozen": "stand_backend_scratch_i150_actor.onnx",
    "moving_model_frozen": "stage306_s2652_transition_head_actor.onnx",
    "signed_pitch_reporting_only": True,
    "expected_policy_slot_counts_per_episode": expected_counts,
    "groups": groups,
    "decision": {
        "all_10_episodes_valid": source["valid"] == 5 and candidate["valid"] == 5,
        "role_contract_pass": role_contract_pass,
        "candidate_not_worse_full_gate": candidate["passes"] >= source["passes"],
        "candidate_improves_full_gate": candidate["passes"] > source["passes"],
        "updates25_unlocked": False,
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
