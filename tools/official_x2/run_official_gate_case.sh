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
DOCKER_IMAGE="${DOCKER_IMAGE:-x2-aimdk-humble:1.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-70}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
RETRY_COOLDOWN_SECONDS="${RETRY_COOLDOWN_SECONDS:-3}"
REPORT_SCENE_XML="${REPORT_SCENE_XML:-}"

mkdir -p "$RESULT_ROOT"

PASS_ENV=(
  COMMAND_VX POLICY_VX_FLOOR PHASE_OFFSET CONTROL_MODE CLOCK_MODE MIRROR_POLICY PD_PROFILE PD_KP_MULTIPLIER PD_KD_MULTIPLIER DEFAULT_POSE_PROFILE
  PREPARE_SECONDS STAND_SECONDS MOVE_SECONDS MOVE_TEMPLATE_MULTIPLIER STOP_SECONDS HEADING_GAIN
  HEADING_RECOVERY_ENTER_RAD HEADING_RECOVERY_EXIT_RAD
  HEADING_ACTION_RECOVERY_ENTER_RAD HEADING_ACTION_RECOVERY_EXIT_RAD
  HEADING_ACTION_RECOVERY_GAIN HEADING_ACTION_RECOVERY_LIMIT
  STATE_PREDICTION_SECONDS STATE_QOS_DEPTH
  CROSS_TRACK_HEADING_GAIN CROSS_TRACK_HEADING_LIMIT HEADING_RATE_LIMIT FIXED_WZ
  ACTION_BIAS_MODE ACTION_BIAS ACTION_BIAS_RAMP_SECONDS ANKLE_ROLL_COMMON_BIAS
  LEFT_HIP_YAW_BIAS RIGHT_HIP_YAW_BIAS YAW_ACTION_GAIN LATERAL_POSITION_GAIN
  TURN_FEEDBACK_FADE_SECONDS
  LATERAL_VELOCITY_GAIN RECOVERY_ENTER_M RECOVERY_EXIT_M RECOVERY_SLEW_RATE_PER_S
  PHASE_ACTION_BOOST ACTION_EMA_ALPHA WAIST_TILT_ACTION_MULTIPLIER STATIONARY_CONTROLLER STOP_CONTROLLER STOP_TRANSITION_SECONDS STOP_BRAKE_GAIN
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
  if [[ -n "$REPORT_SCENE_XML" ]]; then
    [[ -f "$REPORT_SCENE_XML" ]] || { echo "missing REPORT_SCENE_XML: $REPORT_SCENE_XML" >&2; return 2; }
    scene_mount="-v $REPORT_SCENE_XML:/workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml:ro"
  fi
  timeout "${TIMEOUT_SECONDS}s" sg docker -c "docker run --rm --name x2-${CASE_NAME} --network host --ipc host \
    ${DOCKER_ENV[*]} \
    -v $SCRIPT_DIR/run_official_gate_inner.sh:/run_official_gate_inner.sh:ro \
    -v $OFFICIAL_WORKSPACE:/workspace \
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
