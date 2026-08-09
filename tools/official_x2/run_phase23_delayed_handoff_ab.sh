#!/usr/bin/env bash
set -uo pipefail

# BASE Phase23: the only changed variable is the default-off state/support gate.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
MODEL_ROOT="${MODEL_ROOT:-$OFFICIAL_ROOT/models}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/phase23_delayed_handoff_ab.json}"
REPEATS=5

ADAPTER_SHA="a02faa192b2fed9c5cac502019d724a922eaa9261bb1d699a5adafa7ac673cfa"
GATE_CODE_SHA="a710586c7017d028d9bd2f9152033326e14d8bbe3a1767addcbfdb6db94199a0"
GATE_FILE_SHA="a097378247a6cfc1d5f25fcc97a38674576902789c13374bc9a70692c428548e"
HANDOFF_SHA="cd4df30310205a0001f52d1b5e44aae63ccd5ed6f5bd13367a59e09afe2c1a32"
INNER_SHA="883b0f7d3e69ae88de1485b39940f7839b9f756a91170ac088431d4368352cfe"
OUTER_SHA="e5ae5d4e79b766f6d8c3609a1b59bffd8fe70ebefff5213e37411b89e183fc66"
MOVING_SHA="da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c"
STAND_SHA="edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565"
RECOVERY_SHA="5735e4ef29a1ad39ae3ca957e6731efb4593ebdc004ef84856e9adfc8755cd34"

check_sha() {
  local expected="$1" path="$2" actual
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA mismatch: $path actual=$actual expected=$expected" >&2; exit 2; }
}
check_sha "$ADAPTER_SHA" "$SCRIPT_DIR/stage208_official_mujoco_adapter.py"
check_sha "$GATE_CODE_SHA" "$SCRIPT_DIR/handoff_support_gate.py"
check_sha "$GATE_FILE_SHA" "$REPO_ROOT/manifests/x2_phase23_previous_action_support_gate.json"
check_sha "$HANDOFF_SHA" "$SCRIPT_DIR/skill_handoff_contract.py"
check_sha "$INNER_SHA" "$SCRIPT_DIR/run_official_gate_inner.sh"
check_sha "$OUTER_SHA" "$SCRIPT_DIR/run_official_gate_case.sh"
check_sha "$MOVING_SHA" "$MODEL_ROOT/stage306_s2652_transition_head_actor.onnx"
check_sha "$STAND_SHA" "$MODEL_ROOT/stand_backend_scratch_i150_actor.onnx"
check_sha "$RECOVERY_SHA" "$MODEL_ROOT/phase21_outcome_aware_f005_u5_actor.onnx"

labels=(fixed gated)
case_index=0
for label in "${labels[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    case_index=$((case_index + 1))
    case_name="phase23_${label}_handoff_f005_stiff1p2_fixed_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    [[ ! -e "$result" ]] || { echo "refusing to reuse Phase23 result: $result" >&2; exit 2; }
    gate_env=()
    if [[ "$label" == "gated" ]]; then
      gate_env=(
        CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT=/repo/manifests/x2_phase23_previous_action_support_gate.json
        CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT_SHA256="$GATE_FILE_SHA"
      )
    fi
    env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((200 + case_index))" \
      RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
      MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
      STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx \
      RECOVERY_MODEL_PATH=/models/phase21_outcome_aware_f005_u5_actor.onnx \
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
      MAX_ATTEMPTS=2 "${gate_env[@]}" \
      bash "$SCRIPT_DIR/run_official_gate_case.sh" || true
    [[ -f "$result" ]] || { echo "invalid Phase23 episode: missing $result" >&2; exit 3; }
    python3 - "$result" "$label" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8")).get("summary", {})
label = sys.argv[2]
required = ("stand_gate_pass", "startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")
if any(key not in s for key in required):
    raise SystemExit("Phase23 summary/interface mismatch")
counts = s.get("policy_slot_inference_counts", {})
if counts.get("stationary") != 100 or counts.get("main", 0) + counts.get("recovery", 0) != 660:
    raise SystemExit(f"Phase23 slot-count mismatch: {counts}")
enabled = bool(s.get("curriculum_recovery_handoff_gate_enabled"))
if enabled != (label == "gated"):
    raise SystemExit("Phase23 gate enabled-state mismatch")
if label == "fixed" and counts != {"main": 360, "stationary": 100, "recovery": 300}:
    raise SystemExit(f"Phase23 fixed handoff is not tick-equivalent: {counts}")
PY
    [[ "$?" -eq 0 ]] || exit 3
  done
done

PYTHONPATH="$REPO_ROOT/tools" python3 "$SCRIPT_DIR/audit_phase23_delayed_handoff.py" \
  --result-root "$RESULT_ROOT" --output "$SUMMARY_PATH"
echo "Phase23 complete: no training or automatic unlock."
