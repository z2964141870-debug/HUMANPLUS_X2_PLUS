#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CASE_NAME="${CASE_NAME:?CASE_NAME is required}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is required}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
OFFICIAL_WORKSPACE="${OFFICIAL_WORKSPACE:-$OFFICIAL_ROOT/worktree/x2_rl_deploy}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
MODEL_ROOT="${MODEL_ROOT:-$OFFICIAL_ROOT/models}"
TEMPLATE_PATH="${TEMPLATE_PATH:-/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz}"
OFFICIAL_DEFAULT_YAML="${OFFICIAL_DEFAULT_YAML:-$OFFICIAL_ROOT/vendor/aimdk-aarch64-a424add7-artifacts/extra/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml}"
SIMULATOR_DEFAULT_YAML="${SIMULATOR_DEFAULT_YAML:-$OFFICIAL_WORKSPACE/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/simulator/default.yaml}"
SIMULATOR_HTTP_PORT="${SIMULATOR_HTTP_PORT:-}"
DOCKER_IMAGE="${DOCKER_IMAGE:-x2-aimdk-humble:1.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
RETRY_COOLDOWN_SECONDS="${RETRY_COOLDOWN_SECONDS:-3}"
REPORT_SCENE_XML="${REPORT_SCENE_XML:-}"

mkdir -p "$RESULT_ROOT"

SIMULATOR_CONFIG_OVERRIDE=""
cleanup_host() {
  [[ -z "$SIMULATOR_CONFIG_OVERRIDE" ]] || rm -f -- "$SIMULATOR_CONFIG_OVERRIDE"
}
trap cleanup_host EXIT INT TERM

if [[ -n "$SIMULATOR_HTTP_PORT" ]]; then
  [[ "$SIMULATOR_HTTP_PORT" =~ ^[0-9]+$ ]] || {
    echo "SIMULATOR_HTTP_PORT must be numeric" >&2
    exit 2
  }
  (( SIMULATOR_HTTP_PORT >= 1024 && SIMULATOR_HTTP_PORT < 32768 )) || {
    echo "SIMULATOR_HTTP_PORT must be in the non-ephemeral range [1024,32767]" >&2
    exit 2
  }
  [[ -f "$SIMULATOR_DEFAULT_YAML" ]] || {
    echo "missing simulator config: $SIMULATOR_DEFAULT_YAML" >&2
    exit 2
  }
  SIMULATOR_CONFIG_OVERRIDE="$(mktemp /tmp/x2-simulator-default.XXXXXX.yaml)"
  python3 - "$SIMULATOR_DEFAULT_YAML" "$SIMULATOR_CONFIG_OVERRIDE" "$SIMULATOR_HTTP_PORT" <<'PY'
from pathlib import Path
import re
import sys

source, output, port = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
text = source.read_text(encoding="utf-8")
pattern = r"(?m)^(\s*listen_port:\s*)51822(\s*)$"
patched, count = re.subn(pattern, rf"\g<1>{port}\g<2>", text)
if count != 1:
    raise RuntimeError(f"expected exactly one AimRT HTTP port 51822, found {count}")
output.write_text(patched, encoding="utf-8")
PY
fi

# Fail before starting a multi-process official simulation when migration left a
# same-named directory or omitted an immutable baseline input.
[[ -f "$MODEL_ROOT/stage219_s2600_actor.onnx" ]] || {
  echo "missing Stage219 baseline ONNX: $MODEL_ROOT/stage219_s2600_actor.onnx" >&2
  exit 2
}
[[ -f "$TEMPLATE_PATH" ]] || {
  echo "missing gait-template file (directories are invalid): $TEMPLATE_PATH" >&2
  exit 2
}

PASS_ENV=(
  COMMAND_VX POLICY_VX_FLOOR PHASE_OFFSET CONTROL_MODE CLOCK_MODE MIRROR_POLICY ACTOR_SYMMETRY_PROJECTION_ALPHA ACTOR_SYMMETRY_PROJECTION_MASK PD_PROFILE PD_KP_MULTIPLIER PD_KD_MULTIPLIER DEFAULT_POSE_PROFILE
  PREPARE_SECONDS STAND_SECONDS MOVE_SECONDS MOVE_ACCELERATE_SECONDS MOVE_TEMPLATE_MULTIPLIER STOP_SECONDS HEADING_GAIN
  HEADING_RECOVERY_ENTER_RAD HEADING_RECOVERY_EXIT_RAD
  HEADING_ACTION_RECOVERY_ENTER_RAD HEADING_ACTION_RECOVERY_EXIT_RAD
  HEADING_ACTION_RECOVERY_GAIN HEADING_ACTION_RECOVERY_LIMIT
  STATE_PREDICTION_SECONDS STATE_QOS_DEPTH
  CROSS_TRACK_HEADING_GAIN CROSS_TRACK_HEADING_LIMIT HEADING_RATE_LIMIT FIXED_WZ
  ACTION_BIAS_MODE ACTION_BIAS ACTION_BIAS_RAMP_SECONDS ANKLE_ROLL_COMMON_BIAS
  LEFT_HIP_YAW_BIAS RIGHT_HIP_YAW_BIAS YAW_ACTION_GAIN LATERAL_POSITION_GAIN
  TURN_FEEDBACK_FADE_SECONDS
  LATERAL_VELOCITY_GAIN RECOVERY_ENTER_M RECOVERY_EXIT_M RECOVERY_SLEW_RATE_PER_S
  PHASE_ACTION_BOOST ACTION_EMA_ALPHA WAIST_TILT_ACTION_MULTIPLIER STATIONARY_CONTROLLER STOP_CONTROLLER STOP_TRANSITION_SECONDS CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT_SHA256 CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT_SHA256 POST_HANDOFF_SNAPSHOT_OUTPUT POST_HANDOFF_SNAPSHOT_HORIZON_SECONDS STOP_EVENT_SNAPSHOT_OUTPUT STOP_EVENT_SNAPSHOT_V2_OUTPUT STOP_EVENT_SNAPSHOT_HORIZON_SECONDS STOP_BRAKE_GAIN
  STOP_INTENT_DECELERATE_SECONDS
  FUTURE_STOP_PREVIEW_SECONDS
  STOP_BRAKE_LIMIT STOP_BRAKE_TEMPLATE_SPEED STOP_BRAKE_TEMPLATE_FLOOR
  EVENT_HOLD_MIN_SECONDS EVENT_HOLD_SPEED EVENT_HOLD_TILT
  STOP_EMERGENCY_TILT_RAD STOP_EMERGENCY_SPEED_MAX STOP_EMERGENCY_MIN_SECONDS MODEL_PATH
  STATIONARY_MODEL_PATH RECOVERY_MODEL_PATH STATIONARY_WARMUP_SECONDS STATIONARY_BLEND RECORD_X11_TCP
  UPPER_MOTION UPPER_SCALE UPPER_INTENT_SCALE UPPER_START_SECONDS UPPER_TIME_SCALE UPPER_MAX_EXCURSION_RAD
  UPPER_MAX_VELOCITY_RADPS UPPER_FALLBACK_TILT_RAD UPPER_FALLBACK_HEIGHT_M
  UPPER_FALLBACK_HEADING_RAD UPPER_FALLBACK_LATCH UPPER_LOOP UPPER_STOP_MODE
)
DOCKER_ENV=(-e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" -e "CASE_NAME=$CASE_NAME" -e OUTPUT_ROOT=/results)
for name in "${PASS_ENV[@]}"; do
  if [[ -v "$name" ]]; then
    DOCKER_ENV+=(-e "$name=${!name}")
  fi
done

run_container() {
  local scene_mount=""
  local simulator_config_mount=""
  if [[ -n "$REPORT_SCENE_XML" ]]; then
    [[ -f "$REPORT_SCENE_XML" ]] || { echo "missing REPORT_SCENE_XML: $REPORT_SCENE_XML" >&2; return 2; }
    scene_mount="-v $REPORT_SCENE_XML:/workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml:ro"
  fi
  if [[ -n "$SIMULATOR_CONFIG_OVERRIDE" ]]; then
    simulator_config_mount="-v $SIMULATOR_CONFIG_OVERRIDE:/workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/simulator/default.yaml:ro"
  fi
  timeout "${TIMEOUT_SECONDS}s" sg docker -c "docker run --rm --init --name x2-${CASE_NAME} --network host --ipc host \
    ${DOCKER_ENV[*]} \
    -v $SCRIPT_DIR/run_official_gate_inner.sh:/run_official_gate_inner.sh:ro \
    -v $OFFICIAL_WORKSPACE:/workspace \
    $simulator_config_mount \
    -v $REPO_ROOT:/repo:ro \
    -v $MODEL_ROOT:/models:ro \
    -v $TEMPLATE_PATH:/template.npz:ro \
    -v $RESULT_ROOT:/results \
    $scene_mount \
    -v $OFFICIAL_DEFAULT_YAML:/workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml:ro \
    $DOCKER_IMAGE bash -lc 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; bash /run_official_gate_inner.sh'"
}

container_status=1
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if run_container; then
    container_status=0
    break
  else
    container_status=$?
  fi
  echo "official gate infrastructure attempt $attempt/$MAX_ATTEMPTS failed for $CASE_NAME" >&2
  [[ "$attempt" -lt "$MAX_ATTEMPTS" ]] && sleep "$RETRY_COOLDOWN_SECONDS"
done
[[ "$container_status" -eq 0 ]] || exit "$container_status"

python3 - "$RESULT_ROOT/$CASE_NAME.json" <<'PY'
import json, sys
p = sys.argv[1]
s = json.load(open(p, encoding="utf-8"))["summary"]
print(f"{p}: startup={s.get('startup_gate_pass')} move={s.get('move_gate_pass')} stop={s.get('stop_gate_pass')} full={s.get('full_gate_pass')}")
raise SystemExit(0 if s.get("full_gate_pass") else 2)
PY
