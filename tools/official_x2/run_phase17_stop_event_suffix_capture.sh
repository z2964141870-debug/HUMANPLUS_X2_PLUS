#!/usr/bin/env bash
set -euo pipefail

# BASE Phase17: at most five frozen Stage326 mixed-outcome episodes. No training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
PREFIX="phase17_stage326_stop_event_stateful"
MAX_EPISODES=5

check_sha() {
  local expected="$1" path="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "frozen SHA mismatch path=$path actual=$actual expected=$expected" >&2
    exit 2
  }
}

check_sha 61c73512981775c3f7f08fe84e8e6d58e6a76fb7d11ccfb58e4f6665a20d1077 "$SCRIPT_DIR/stage208_official_mujoco_adapter.py"
check_sha c4868fdc266114818cccbd04413eaef747bbd6b7c39b89d9a355e39a74d099a8 "$SCRIPT_DIR/stop_event_suffix_contract.py"
check_sha b0e46dfdd6ff7597c6522a267fc259d9dffd1c520cb2cc872822c0aa4d8e7182 "$SCRIPT_DIR/controller_snapshot_contract.py"
check_sha 20a52e6b88aa815739d9329dbebaff0acf62fea1ee51b8b9b29aab1fd3d61e91 "$SCRIPT_DIR/run_official_gate_inner.sh"
check_sha c4507237a4d1256db63995a9e1f52cb1c386e98586fcc7b8d4c0fd3603784f6e "$SCRIPT_DIR/run_official_gate_case.sh"
check_sha 84a7cf6a041499c4220de85e2a8103a052d79404708089b59d19e5c3343c74c3 "$SCRIPT_DIR/validate_phase17_stop_event_episode.py"
check_sha da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c "$OFFICIAL_ROOT/models/stage306_s2652_transition_head_actor.onnx"
check_sha edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565 "$OFFICIAL_ROOT/models/stand_backend_scratch_i150_actor.onnx"

success=0
failure=0
critical=0
completed=0
for episode in $(seq 1 "$MAX_EPISODES"); do
  case_name="${PREFIX}_r${episode}"
  trace="$RESULT_ROOT/${case_name}.json"
  sidecar="$RESULT_ROOT/${case_name}_sidecar.json"
  if [[ -e "$trace" && ! -e "$sidecar" ]] || [[ ! -e "$trace" && -e "$sidecar" ]]; then
    echo "incomplete pre-existing Phase17 output pair: $case_name" >&2
    exit 2
  fi
  if [[ ! -e "$trace" ]]; then
    env \
    CASE_NAME="$case_name" ROS_DOMAIN_ID="$((176 + episode))" \
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
    STOP_EVENT_SNAPSHOT_OUTPUT="/results/${case_name}_sidecar.json" \
    STOP_EVENT_SNAPSHOT_HORIZON_SECONDS=4.0 \
    MAX_ATTEMPTS=1 TIMEOUT_SECONDS=180 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh" || true
  else
    echo "resume: validating and counting existing immutable Phase17 pair $case_name"
  fi

  [[ -f "$trace" && -f "$sidecar" ]] || {
    echo "Phase17 episode missing trace/sidecar: $case_name" >&2
    exit 2
  }
  classification="$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT/tools" python3 \
    "$SCRIPT_DIR/validate_phase17_stop_event_episode.py" "$trace" "$sidecar")"
  case "$classification" in
    success) success=$((success + 1)) ;;
    failure) failure=$((failure + 1)) ;;
    critical) critical=$((critical + 1)) ;;
    *) echo "invalid Phase17 classification: $classification" >&2; exit 2 ;;
  esac
  completed=$((completed + 1))
  echo "Phase17 episode=$completed class=$classification cumulative success=$success critical=$critical failure=$failure"
  if [[ "$success" -ge 2 && "$failure" -ge 1 ]]; then
    echo "Phase17 outcome target met; stopping early at $completed episodes"
    break
  fi
done

python3 - "$RESULT_ROOT" "$PREFIX" "$completed" "$success" "$critical" "$failure" <<'PY'
import json, pathlib, sys
root, prefix = pathlib.Path(sys.argv[1]), sys.argv[2]
completed, success, critical, failure = map(int, sys.argv[3:])
rows=[]
for episode in range(1, completed+1):
    trace=root/f"{prefix}_r{episode}.json"
    sidecar=root/f"{prefix}_r{episode}_sidecar.json"
    summary=json.loads(trace.read_text(encoding="utf-8"))["summary"]
    rows.append({
        "episode":episode, "trace":str(trace), "sidecar":str(sidecar),
        "full_gate_pass":bool(summary.get("full_gate_pass")),
        "stop_gate_pass":bool(summary.get("stop_gate_pass")),
        "survived_stop_height_gate":bool(summary.get("survived_stop_height_gate")),
    })
print(json.dumps({
    "stage":"BASE Phase17 stop-event stateful suffix capture",
    "completed_episodes":completed,
    "counts":{"success":success,"critical":critical,"failure":failure},
    "target_met":success>=2 and failure>=1,
    "training_unlocked":False,
    "rows":rows,
}, indent=2))
PY
