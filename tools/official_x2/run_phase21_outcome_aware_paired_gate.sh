#!/usr/bin/env bash
set -uo pipefail

# BASE Phase21 frozen official AimDK matched-event gate. Moving and stationary
# actors are identical in all 15 episodes. Only the post-handoff recovery slot
# differs among source, same-seed fraction-0 control, and fraction-0.05 candidate.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
MODEL_ROOT="${MODEL_ROOT:-$OFFICIAL_ROOT/models}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/phase21_outcome_aware_paired_gate.json}"
REPEATS=5

ADAPTER_SHA="6b7c6c353727c75877938372ef2240b8e87e45ad967abb361aacce2e5f66f9d9"
HANDOFF_SHA="cd4df30310205a0001f52d1b5e44aae63ccd5ed6f5bd13367a59e09afe2c1a32"
INNER_SHA="d93b36b750f8d6ea76718e9cae049156bc6f9eb9ab203667b77a24095b70aa06"
OUTER_SHA="dc590b2a2b0080fa5e88cd13124f7b467bd56e70c68a1804fb0e3c4e84d06fd0"
MOVING_SHA="da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c"
SOURCE_SHA="edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565"
F000_SHA="20e1859a36760bf32f0c29c94c491c7ac6fab86f8fddd3b6fbe769ba3e636449"
F005_SHA="5735e4ef29a1ad39ae3ca957e6731efb4593ebdc004ef84856e9adfc8755cd34"

check_sha() {
  local expected="$1" path="$2" actual
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA mismatch: $path actual=$actual expected=$expected" >&2
    exit 2
  }
}
check_sha "$ADAPTER_SHA" "$SCRIPT_DIR/stage208_official_mujoco_adapter.py"
check_sha "$HANDOFF_SHA" "$SCRIPT_DIR/skill_handoff_contract.py"
check_sha "$INNER_SHA" "$SCRIPT_DIR/run_official_gate_inner.sh"
check_sha "$OUTER_SHA" "$SCRIPT_DIR/run_official_gate_case.sh"
check_sha "$MOVING_SHA" "$MODEL_ROOT/stage306_s2652_transition_head_actor.onnx"
check_sha "$SOURCE_SHA" "$MODEL_ROOT/stand_backend_scratch_i150_actor.onnx"
check_sha "$F000_SHA" "$MODEL_ROOT/phase21_outcome_aware_f000_u5_actor.onnx"
check_sha "$F005_SHA" "$MODEL_ROOT/phase21_outcome_aware_f005_u5_actor.onnx"

labels=(source f000 f005)
recovery_models=(
  stand_backend_scratch_i150_actor.onnx
  phase21_outcome_aware_f000_u5_actor.onnx
  phase21_outcome_aware_f005_u5_actor.onnx
)

case_index=0
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  recovery="${recovery_models[$index]}"
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="phase21_outcome_aware_${label}_matched_blend0p5_stiff1p2_fixed_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    [[ ! -e "$result" ]] || { echo "refusing to reuse Phase21 result: $result" >&2; exit 2; }
    env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((180 + case_index))" \
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
      CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS=0.5 \
      FUTURE_STOP_PREVIEW_SECONDS=0.5 \
      ACTION_BIAS_MODE=none ACTION_BIAS=0 ANKLE_ROLL_COMMON_BIAS=0 \
      LEFT_HIP_YAW_BIAS=0 RIGHT_HIP_YAW_BIAS=0 YAW_ACTION_GAIN=0 \
      LATERAL_POSITION_GAIN=0 LATERAL_VELOCITY_GAIN=0 HEADING_GAIN=0 \
      CROSS_TRACK_HEADING_GAIN=0 STATE_PREDICTION_SECONDS=0 STATE_QOS_DEPTH=10 \
      ACTION_EMA_ALPHA=1 WAIST_TILT_ACTION_MULTIPLIER=1 \
      PD_PROFILE=official_kp_ankle PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2 \
      MAX_ATTEMPTS=2 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh" || true
    [[ -f "$result" ]] || { echo "invalid Phase21 episode: missing $result" >&2; exit 3; }
    python3 - "$result" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
s = p.get("summary", {})
needed = ("stand_gate_pass", "startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")
if any(key not in s for key in needed):
    raise SystemExit("invalid Phase21 summary/interface")
if s.get("policy_slot_inference_counts") != {"main": 360, "stationary": 100, "recovery": 300}:
    raise SystemExit("Phase21 policy-slot contract mismatch")
PY
    [[ "$?" -eq 0 ]] || exit 3
  done
done

python3 - "$RESULT_ROOT" "$SUMMARY_PATH" "${recovery_models[@]}" <<'PY'
import hashlib, json, math, pathlib, statistics, sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
models = dict(zip(("source", "f000", "f005"), sys.argv[3:6]))
model_root = pathlib.Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models")
expected_counts = {"main": 360, "stationary": 100, "recovery": 300}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def row(label, repeat):
    path = root / f"phase21_outcome_aware_{label}_matched_blend0p5_stiff1p2_fixed_r{repeat}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    s = payload["summary"]
    return {
        "path": str(path),
        "file_sha256": sha(path),
        "valid": all(k in s for k in ("stand_gate_pass", "startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")),
        "stand": bool(s.get("stand_gate_pass")),
        "startup": bool(s.get("startup_gate_pass")),
        "move": bool(s.get("move_gate_pass")),
        "stop": bool(s.get("stop_gate_pass")),
        "full": bool(s.get("full_gate_pass")),
        "heading_max_rad": s.get("move_heading_max_deviation_rad"),
        "lateral_displacement_m": s.get("move_lateral_displacement_m"),
        "stop_drift_m": s.get("stop_root_xy_drift_m"),
        "stop_settle_time_s": s.get("stop_settle_time_s"),
        "root_pitch_mean_rad": {phase: s.get(f"{phase}_root_pitch_mean_rad") for phase in ("stand", "startup", "move", "stop")},
        "policy_slot_counts": s.get("policy_slot_inference_counts"),
    }

def aggregate(rows):
    continuous = ("heading_max_rad", "lateral_displacement_m", "stop_drift_m", "stop_settle_time_s")
    return {
        "valid": sum(r["valid"] for r in rows),
        "subgates": {key: sum(r[key] for r in rows) for key in ("stand", "startup", "move", "stop", "full")},
        "metrics": {
            key: {
                "mean": statistics.fmean(float(r[key]) for r in rows),
                "median": statistics.median(float(r[key]) for r in rows),
                "population_std": statistics.pstdev(float(r[key]) for r in rows),
            }
            for key in continuous
        },
    }

groups = {}
for label in ("source", "f000", "f005"):
    rows = [row(label, repeat) for repeat in range(1, 6)]
    groups[label] = {
        "recovery_model": models[label],
        "recovery_model_sha256": sha(model_root / models[label]),
        "rows": rows,
        "aggregate": aggregate(rows),
    }

def candidate_not_below(candidate, reference, key):
    return candidate["aggregate"]["subgates"][key] >= reference["aggregate"]["subgates"][key]

source, control, candidate = (groups[name] for name in ("source", "f000", "f005"))
subgate_not_worse = {
    key: candidate_not_below(candidate, source, key) and candidate_not_below(candidate, control, key)
    for key in ("stand", "startup", "move", "stop", "full")
}
all_valid = all(group["aggregate"]["valid"] == 5 for group in groups.values())
slots_exact = all(r["policy_slot_counts"] == expected_counts for g in groups.values() for r in g["rows"])
result = {
    "stage": "BASE Phase21 outcome-aware paired 5-update official gate",
    "hypothesis": "A 5% balanced outcome-aware state-role reset curriculum improves recovery without regressing source or same-seed fraction-0 control.",
    "intervention": "same source/seed/64env/5updates; fraction 0.00 -> 0.05 only",
    "official_contract": {
        "moving": "frozen Stage306",
        "stationary": "frozen source i150",
        "recovery": "source vs f000 vs f005",
        "handoff_blend_s": 0.5,
        "stiff_fixed_multiplier": 1.2,
        "fixed_upper": True,
        "expected_policy_slot_counts": expected_counts,
    },
    "groups": groups,
    "decision": {
        "all_15_valid": all_valid,
        "all_policy_slots_exact": slots_exact,
        "candidate_subgates_not_worse_than_source_and_f000": subgate_not_worse,
        "candidate_not_worse": all(subgate_not_worse.values()),
        "candidate_moves_recovery_toward_5_of_5": candidate["aggregate"]["subgates"]["stop"] > max(source["aggregate"]["subgates"]["stop"], control["aggregate"]["subgates"]["stop"]) or candidate["aggregate"]["subgates"]["full"] > max(source["aggregate"]["subgates"]["full"], control["aggregate"]["subgates"]["full"]),
        "phase21_promoted": False,
        "updates25_unlocked": False,
    },
    "diagnostic_only": {"training_reward": True, "episode_length": True, "signed_pitch": True},
}
result["decision"]["phase21_promoted"] = bool(
    all_valid and slots_exact and all(subgate_not_worse.values()) and result["decision"]["candidate_moves_recovery_toward_5_of_5"]
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY

echo "Phase21 official panel complete. This runner never unlocks 25 updates automatically."
