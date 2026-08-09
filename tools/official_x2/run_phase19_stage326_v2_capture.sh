#!/usr/bin/env bash
set -euo pipefail

# BASE Phase19: at most five frozen Stage326 episodes, recorder v2 only. No training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
PREFIX="phase19_stage326_stop_event_stateful_v2"
MAX_EPISODES=5

check_sha() {
  local expected="$1" path="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "frozen SHA mismatch path=$path actual=$actual expected=$expected" >&2
    exit 2
  }
}

check_sha 6b7c6c353727c75877938372ef2240b8e87e45ad967abb361aacce2e5f66f9d9 "$SCRIPT_DIR/stage208_official_mujoco_adapter.py"
check_sha eae96b1e0ebd8b35379df8f2fb8d9b0a45aff4bb376141851af60ea083dbdbdb "$SCRIPT_DIR/stop_event_suffix_contract_v2.py"
check_sha c4868fdc266114818cccbd04413eaef747bbd6b7c39b89d9a355e39a74d099a8 "$SCRIPT_DIR/stop_event_suffix_contract.py"
check_sha b0e46dfdd6ff7597c6522a267fc259d9dffd1c520cb2cc872822c0aa4d8e7182 "$SCRIPT_DIR/controller_snapshot_contract.py"
check_sha d93b36b750f8d6ea76718e9cae049156bc6f9eb9ab203667b77a24095b70aa06 "$SCRIPT_DIR/run_official_gate_inner.sh"
check_sha dc590b2a2b0080fa5e88cd13124f7b467bd56e70c68a1804fb0e3c4e84d06fd0 "$SCRIPT_DIR/run_official_gate_case.sh"
check_sha ab2b7549a47759dff24daff46718532c0941e5e4f40b126698e0c4f009cc17a0 "$SCRIPT_DIR/validate_phase19_stop_event_episode_v2.py"
check_sha da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c "$OFFICIAL_ROOT/models/stage306_s2652_transition_head_actor.onnx"
check_sha edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565 "$OFFICIAL_ROOT/models/stand_backend_scratch_i150_actor.onnx"

success=0
height_failure=0
critical=0
invalid_artifact=0
completed=0
for episode in $(seq 1 "$MAX_EPISODES"); do
  case_name="${PREFIX}_r${episode}"
  trace="$RESULT_ROOT/${case_name}.json"
  sidecar="$RESULT_ROOT/${case_name}_sidecar_v2.json"
  if [[ "$episode" -eq 1 && -e "$trace" && ! -e "$sidecar" ]]; then
    # Pre-registered first attempt completed physics and 201 row-level checks,
    # then failed final sidecar hashing on an Infinity in the full summary.
    # Preserve it as an invalid artifact; never overwrite, rerun, or classify.
    invalid_artifact=$((invalid_artifact + 1))
    completed=$((completed + 1))
    echo "Phase19 r1 preserved as invalid trace-only artifact; continuing at r2"
    continue
  fi
  if [[ -e "$trace" || -e "$sidecar" ]]; then
    echo "Phase19 outputs are immutable; refusing pre-existing pair: $case_name" >&2
    exit 2
  fi
  env \
    CASE_NAME="$case_name" ROS_DOMAIN_ID="$((196 + episode))" \
    RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$OFFICIAL_ROOT/models" \
    MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
    STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx \
    COMMAND_VX=0.30 POLICY_VX_FLOOR=0 PHASE_OFFSET=0 CLOCK_MODE=step \
    CONTROL_MODE=full DEFAULT_POSE_PROFILE=stage208 \
    PREPARE_SECONDS=0.2 STAND_SECONDS=2.0 STATIONARY_CONTROLLER=policy \
    STATIONARY_WARMUP_SECONDS=0 STATIONARY_BLEND=0.5 \
    MOVE_SECONDS=4.0 MOVE_ACCELERATE_SECONDS=0 MOVE_TEMPLATE_MULTIPLIER=1.0 \
    STOP_SECONDS=8.0 STOP_CONTROLLER=brake_blend_to_policy \
    STOP_TRANSITION_SECONDS=1.0 STOP_INTENT_DECELERATE_SECONDS=2.0 \
    FUTURE_STOP_PREVIEW_SECONDS=0.5 STOP_BRAKE_GAIN=1.5 \
    EVENT_HOLD_MIN_SECONDS=0.5 EVENT_HOLD_SPEED=0.05 \
    ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.6 \
    ANKLE_ROLL_COMMON_BIAS=0.20 \
    LATERAL_POSITION_GAIN=0.8 LATERAL_VELOCITY_GAIN=0.2 \
    RECOVERY_ENTER_M=0.08 RECOVERY_EXIT_M=0.03 RECOVERY_SLEW_RATE_PER_S=1.0 \
    ACTION_EMA_ALPHA=1 WAIST_TILT_ACTION_MULTIPLIER=1 \
    PD_PROFILE=official_kp_ankle PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2 \
    STOP_EVENT_SNAPSHOT_V2_OUTPUT="/results/${case_name}_sidecar_v2.json" \
    STOP_EVENT_SNAPSHOT_HORIZON_SECONDS=4.0 \
    MAX_ATTEMPTS=1 TIMEOUT_SECONDS=180 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh" || true

  [[ -f "$trace" && -f "$sidecar" ]] || {
    echo "Phase19 episode missing trace/v2 sidecar: $case_name" >&2
    exit 2
  }
  validation="$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT/tools" python3 \
    "$SCRIPT_DIR/validate_phase19_stop_event_episode_v2.py" "$trace" "$sidecar")"
  classification="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["classification"])' "$validation")"
  case "$classification" in
    success) success=$((success + 1)) ;;
    height_failure) height_failure=$((height_failure + 1)) ;;
    critical) critical=$((critical + 1)) ;;
    *) echo "invalid Phase19 classification: $classification" >&2; exit 2 ;;
  esac
  completed=$((completed + 1))
  echo "Phase19 episode=$completed class=$classification cumulative success=$success critical=$critical height_failure=$height_failure"
  if [[ "$success" -ge 1 && "$critical" -ge 1 && "$height_failure" -ge 1 ]]; then
    echo "Phase19 outcome target met; stopping early at $completed episodes"
    break
  fi
done

python3 - "$completed" "$success" "$critical" "$height_failure" "$invalid_artifact" <<'PY'
import json, sys
completed, success, critical, height_failure, invalid_artifact = map(int, sys.argv[1:])
print(json.dumps({
    "stage": "BASE Phase19 recorder-v2 outcome capture",
    "completed_episodes": completed,
    "counts": {"success": success, "critical": critical, "height_failure": height_failure},
    "invalid_artifact_count": invalid_artifact,
    "target_met": success >= 1 and critical >= 1 and height_failure >= 1,
    "training_unlocked": False,
}, indent=2))
PY
