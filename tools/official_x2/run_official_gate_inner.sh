#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
SIM="$ROOT/x2_rl_deploy_mujoco"
CASE_NAME="${CASE_NAME:?CASE_NAME is required}"
COMMAND_VX="${COMMAND_VX:-0.30}"
POLICY_VX_FLOOR="${POLICY_VX_FLOOR:-0.0}"
PHASE_OFFSET="${PHASE_OFFSET:-0.0}"
CONTROL_MODE="${CONTROL_MODE:-full}"
CLOCK_MODE="${CLOCK_MODE:-step}"
MIRROR_POLICY="${MIRROR_POLICY:-false}"
ACTOR_SYMMETRY_PROJECTION_ALPHA="${ACTOR_SYMMETRY_PROJECTION_ALPHA:-0.0}"
ACTOR_SYMMETRY_PROJECTION_MASK="${ACTOR_SYMMETRY_PROJECTION_MASK:-roll_yaw}"
PD_PROFILE="${PD_PROFILE:-official_kp_ankle}"
PD_KP_MULTIPLIER="${PD_KP_MULTIPLIER:-1.0}"
PD_KD_MULTIPLIER="${PD_KD_MULTIPLIER:-1.0}"
DEFAULT_POSE_PROFILE="${DEFAULT_POSE_PROFILE:-stage208}"
PREPARE_SECONDS="${PREPARE_SECONDS:-0.2}"
STAND_SECONDS="${STAND_SECONDS:-2.0}"
MOVE_SECONDS="${MOVE_SECONDS:-4.0}"
MOVE_ACCELERATE_SECONDS="${MOVE_ACCELERATE_SECONDS:-0.0}"
MOVE_TEMPLATE_MULTIPLIER="${MOVE_TEMPLATE_MULTIPLIER:-1.0}"
STOP_SECONDS="${STOP_SECONDS:-8.0}"
STATE_PREDICTION_SECONDS="${STATE_PREDICTION_SECONDS:-0.0}"
STATE_QOS_DEPTH="${STATE_QOS_DEPTH:-10}"
HEADING_GAIN="${HEADING_GAIN:-0.0}"
HEADING_RECOVERY_ENTER_RAD="${HEADING_RECOVERY_ENTER_RAD:-}"
HEADING_RECOVERY_EXIT_RAD="${HEADING_RECOVERY_EXIT_RAD:-0.08}"
HEADING_ACTION_RECOVERY_ENTER_RAD="${HEADING_ACTION_RECOVERY_ENTER_RAD:-}"
HEADING_ACTION_RECOVERY_EXIT_RAD="${HEADING_ACTION_RECOVERY_EXIT_RAD:-0.10}"
HEADING_ACTION_RECOVERY_GAIN="${HEADING_ACTION_RECOVERY_GAIN:-1.0}"
HEADING_ACTION_RECOVERY_LIMIT="${HEADING_ACTION_RECOVERY_LIMIT:-0.25}"
CROSS_TRACK_HEADING_GAIN="${CROSS_TRACK_HEADING_GAIN:-0.0}"
CROSS_TRACK_HEADING_LIMIT="${CROSS_TRACK_HEADING_LIMIT:-0.30}"
HEADING_RATE_LIMIT="${HEADING_RATE_LIMIT:-0.5}"
FIXED_WZ="${FIXED_WZ:-}"
ACTION_BIAS_MODE="${ACTION_BIAS_MODE:-none}"
ACTION_BIAS="${ACTION_BIAS:-0.0}"
ACTION_BIAS_RAMP_SECONDS="${ACTION_BIAS_RAMP_SECONDS:-0.0}"
ANKLE_ROLL_COMMON_BIAS="${ANKLE_ROLL_COMMON_BIAS:-0.0}"
LEFT_HIP_YAW_BIAS="${LEFT_HIP_YAW_BIAS:-0.0}"
RIGHT_HIP_YAW_BIAS="${RIGHT_HIP_YAW_BIAS:-0.0}"
YAW_ACTION_GAIN="${YAW_ACTION_GAIN:-0.0}"
TURN_FEEDBACK_FADE_SECONDS="${TURN_FEEDBACK_FADE_SECONDS:-0.0}"
LATERAL_POSITION_GAIN="${LATERAL_POSITION_GAIN:-0.8}"
LATERAL_VELOCITY_GAIN="${LATERAL_VELOCITY_GAIN:-0.2}"
RECOVERY_ENTER_M="${RECOVERY_ENTER_M:-0.12}"
RECOVERY_EXIT_M="${RECOVERY_EXIT_M:-0.04}"
RECOVERY_SLEW_RATE_PER_S="${RECOVERY_SLEW_RATE_PER_S:-1.0}"
PHASE_ACTION_BOOST="${PHASE_ACTION_BOOST:-0.0}"
ACTION_EMA_ALPHA="${ACTION_EMA_ALPHA:-1.0}"
WAIST_TILT_ACTION_MULTIPLIER="${WAIST_TILT_ACTION_MULTIPLIER:-1.0}"
STATIONARY_CONTROLLER="${STATIONARY_CONTROLLER:-policy}"
STOP_CONTROLLER="${STOP_CONTROLLER:-policy}"
STOP_TRANSITION_SECONDS="${STOP_TRANSITION_SECONDS:-1.0}"
CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS="${CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS:-0.0}"
CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT="${CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT:-}"
CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT_SHA256="${CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT_SHA256:-}"
CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT="${CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT:-}"
CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT_SHA256="${CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT_SHA256:-}"
POST_HANDOFF_SNAPSHOT_OUTPUT="${POST_HANDOFF_SNAPSHOT_OUTPUT:-}"
POST_HANDOFF_SNAPSHOT_HORIZON_SECONDS="${POST_HANDOFF_SNAPSHOT_HORIZON_SECONDS:-1.5}"
STOP_EVENT_SNAPSHOT_OUTPUT="${STOP_EVENT_SNAPSHOT_OUTPUT:-}"
STOP_EVENT_SNAPSHOT_V2_OUTPUT="${STOP_EVENT_SNAPSHOT_V2_OUTPUT:-}"
STOP_EVENT_SNAPSHOT_HORIZON_SECONDS="${STOP_EVENT_SNAPSHOT_HORIZON_SECONDS:-4.0}"
STOP_INTENT_DECELERATE_SECONDS="${STOP_INTENT_DECELERATE_SECONDS:-2.0}"
FUTURE_STOP_PREVIEW_SECONDS="${FUTURE_STOP_PREVIEW_SECONDS:-0.0}"
STOP_BRAKE_GAIN="${STOP_BRAKE_GAIN:-0.8}"
STOP_BRAKE_LIMIT="${STOP_BRAKE_LIMIT:-0.30}"
STOP_BRAKE_TEMPLATE_SPEED="${STOP_BRAKE_TEMPLATE_SPEED:-0.30}"
STOP_BRAKE_TEMPLATE_FLOOR="${STOP_BRAKE_TEMPLATE_FLOOR:-0.25}"
EVENT_HOLD_MIN_SECONDS="${EVENT_HOLD_MIN_SECONDS:-0.5}"
EVENT_HOLD_SPEED="${EVENT_HOLD_SPEED:-0.05}"
EVENT_HOLD_TILT="${EVENT_HOLD_TILT:-0.10}"
STOP_EMERGENCY_TILT_RAD="${STOP_EMERGENCY_TILT_RAD:-}"
STOP_EMERGENCY_SPEED_MAX="${STOP_EMERGENCY_SPEED_MAX:-0.10}"
STOP_EMERGENCY_MIN_SECONDS="${STOP_EMERGENCY_MIN_SECONDS:-0.80}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/results}"
MODEL_PATH="${MODEL_PATH:-/models/stage219_s2600_actor.onnx}"
STATIONARY_MODEL_PATH="${STATIONARY_MODEL_PATH:-/models/stand_backend_scratch_i150_actor.onnx}"
RECOVERY_MODEL_PATH="${RECOVERY_MODEL_PATH:-}"
STATIONARY_WARMUP_SECONDS="${STATIONARY_WARMUP_SECONDS:-0.0}"
STATIONARY_BLEND="${STATIONARY_BLEND:-0.50}"
UPPER_MOTION="${UPPER_MOTION:-}"
UPPER_SCALE="${UPPER_SCALE:-0.25}"
UPPER_INTENT_SCALE="${UPPER_INTENT_SCALE:-}"
UPPER_START_SECONDS="${UPPER_START_SECONDS:-0.0}"
UPPER_TIME_SCALE="${UPPER_TIME_SCALE:-0.5}"
UPPER_MAX_EXCURSION_RAD="${UPPER_MAX_EXCURSION_RAD:-0.12}"
UPPER_MAX_VELOCITY_RADPS="${UPPER_MAX_VELOCITY_RADPS:-0.20}"
UPPER_FALLBACK_TILT_RAD="${UPPER_FALLBACK_TILT_RAD:-0.35}"
UPPER_FALLBACK_HEIGHT_M="${UPPER_FALLBACK_HEIGHT_M:-0.58}"
UPPER_FALLBACK_HEADING_RAD="${UPPER_FALLBACK_HEADING_RAD:-inf}"
UPPER_FALLBACK_LATCH="${UPPER_FALLBACK_LATCH:-false}"
UPPER_LOOP="${UPPER_LOOP:-false}"
UPPER_STOP_MODE="${UPPER_STOP_MODE:-return}"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# The adapter is launched as a file under /repo/tools/official_x2.  Python then
# puts that leaf directory (rather than /repo/tools) on sys.path, so sibling
# package imports such as ``official_x2.skill_handoff_contract`` fail inside
# the otherwise healthy official container.  Make the mounted tools package
# explicit; preserve any image-provided path for ROS/AimDK helpers.
export PYTHONPATH="/repo/tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/install/aimdk_msgs/lib:$SIM/bin:$SIM/lib:/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/onnxruntime/lib"
export SIM_ROBOT_PATH="$SIM/configuration/robot/lx2501_3_t2d5"
export SIM_RESOURCE_MODEL_PATH="$SIM/resource/model"
export XDG_RUNTIME_DIR=/tmp/runtime-root
export AGIBOT_ENABLE_HDS_COMPONENT=false
export AGIBOT_ENABLE_EVENT_COMPONENT=false
export AGIBOT_ENABLE_AUDIT_COMPONENT=false
export AGIBOT_ENABLE_MONITOR=true
export EM_APP_NAME=sim
export AIMRT_PLUGIN_SHM_DEFAULT_WAIT_TIME_US=300

LOG_ROOT="$OUTPUT_ROOT/logs/$CASE_NAME"
mkdir -p "$SIM/bin/log/crash" "$SIM/bin/cfg/tmp" "$XDG_RUNTIME_DIR" "$LOG_ROOT" "$OUTPUT_ROOT"
chmod 700 "$XDG_RUNTIME_DIR"
cleanup() {
  kill "${ADAPTER_PID:-}" "${SIM_PID:-}" "${XVFB_PID:-}" 2>/dev/null || true
  wait "${ADAPTER_PID:-}" "${SIM_PID:-}" "${XVFB_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

XVFB_EXTRA_ARGS=()
if [[ "${RECORD_X11_TCP:-false}" == "true" ]]; then
  # Recording only: expose this throw-away virtual display on the host-networked
  # container so the host ffmpeg process can capture the official MuJoCo viewer.
  XVFB_EXTRA_ARGS=(-ac -listen tcp)
fi
Xvfb :99 -screen 0 1280x720x24 "${XVFB_EXTRA_ARGS[@]}" >"$LOG_ROOT/xvfb.log" 2>&1 &
XVFB_PID=$!
export DISPLAY=:99

EXTRA_ARGS=()
[[ -n "$FIXED_WZ" ]] && EXTRA_ARGS+=(--fixed-wz "$FIXED_WZ")
[[ -n "$HEADING_RECOVERY_ENTER_RAD" ]] && EXTRA_ARGS+=(--heading-recovery-enter-rad "$HEADING_RECOVERY_ENTER_RAD")
[[ -n "$HEADING_ACTION_RECOVERY_ENTER_RAD" ]] && EXTRA_ARGS+=(--heading-action-recovery-enter-rad "$HEADING_ACTION_RECOVERY_ENTER_RAD")
[[ -n "$STATIONARY_MODEL_PATH" ]] && EXTRA_ARGS+=(--stationary-model "$STATIONARY_MODEL_PATH")
[[ -n "$RECOVERY_MODEL_PATH" ]] && EXTRA_ARGS+=(--recovery-model "$RECOVERY_MODEL_PATH")
[[ -n "$CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT" ]] && EXTRA_ARGS+=(
  --curriculum-recovery-handoff-gate-contract "$CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT"
  --curriculum-recovery-handoff-gate-contract-sha256 "$CURRICULUM_RECOVERY_HANDOFF_GATE_CONTRACT_SHA256"
)
[[ -n "$CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT" ]] && EXTRA_ARGS+=(
  --curriculum-recovery-physical-gate-contract "$CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT"
  --curriculum-recovery-physical-gate-contract-sha256 "$CURRICULUM_RECOVERY_PHYSICAL_GATE_CONTRACT_SHA256"
)
[[ "$MIRROR_POLICY" == "true" ]] && EXTRA_ARGS+=(--mirror-policy)
[[ -n "$UPPER_MOTION" ]] && EXTRA_ARGS+=(--upper-motion "$UPPER_MOTION")
[[ -n "$UPPER_INTENT_SCALE" ]] && EXTRA_ARGS+=(--upper-intent-scale "$UPPER_INTENT_SCALE")
[[ -n "$STOP_EMERGENCY_TILT_RAD" ]] && EXTRA_ARGS+=(--stop-emergency-tilt-rad "$STOP_EMERGENCY_TILT_RAD")
[[ -n "$POST_HANDOFF_SNAPSHOT_OUTPUT" ]] && EXTRA_ARGS+=(
  --post-handoff-snapshot-output "$POST_HANDOFF_SNAPSHOT_OUTPUT"
  --post-handoff-snapshot-horizon-seconds "$POST_HANDOFF_SNAPSHOT_HORIZON_SECONDS"
)
[[ -n "$STOP_EVENT_SNAPSHOT_OUTPUT" ]] && EXTRA_ARGS+=(
  --stop-event-snapshot-output "$STOP_EVENT_SNAPSHOT_OUTPUT"
  --stop-event-snapshot-horizon-seconds "$STOP_EVENT_SNAPSHOT_HORIZON_SECONDS"
)
[[ -n "$STOP_EVENT_SNAPSHOT_V2_OUTPUT" ]] && EXTRA_ARGS+=(
  --stop-event-snapshot-v2-output "$STOP_EVENT_SNAPSHOT_V2_OUTPUT"
  --stop-event-snapshot-horizon-seconds "$STOP_EVENT_SNAPSHOT_HORIZON_SECONDS"
)
[[ "$UPPER_LOOP" == "true" ]] && EXTRA_ARGS+=(--upper-loop)
[[ "$UPPER_FALLBACK_LATCH" == "true" ]] && EXTRA_ARGS+=(--upper-fallback-latch)

cd /repo
python3 tools/official_x2/stage208_official_mujoco_adapter.py \
  --model "$MODEL_PATH" --template /template.npz \
  --output "${OUTPUT_ROOT}/${CASE_NAME}.json" --vx "$COMMAND_VX" \
  --policy-vx-floor "$POLICY_VX_FLOOR" \
  --phase-offset "$PHASE_OFFSET" --clock-mode "$CLOCK_MODE" --control-mode "$CONTROL_MODE" \
  --actor-symmetry-projection-alpha "$ACTOR_SYMMETRY_PROJECTION_ALPHA" \
  --actor-symmetry-projection-mask "$ACTOR_SYMMETRY_PROJECTION_MASK" \
  --pd-profile "$PD_PROFILE" --pd-kp-multiplier "$PD_KP_MULTIPLIER" \
  --pd-kd-multiplier "$PD_KD_MULTIPLIER" --default-pose-profile "$DEFAULT_POSE_PROFILE" \
  --prepare-seconds "$PREPARE_SECONDS" --stand-seconds "$STAND_SECONDS" \
  --stationary-warmup-seconds "$STATIONARY_WARMUP_SECONDS" --stationary-blend "$STATIONARY_BLEND" \
  --move-seconds "$MOVE_SECONDS" --stop-seconds "$STOP_SECONDS" \
  --move-accelerate-seconds "$MOVE_ACCELERATE_SECONDS" \
  --move-template-multiplier "$MOVE_TEMPLATE_MULTIPLIER" \
  --state-prediction-seconds "$STATE_PREDICTION_SECONDS" \
  --state-qos-depth "$STATE_QOS_DEPTH" \
  --heading-gain "$HEADING_GAIN" --cross-track-heading-gain "$CROSS_TRACK_HEADING_GAIN" \
  --heading-recovery-exit-rad "$HEADING_RECOVERY_EXIT_RAD" \
  --heading-action-recovery-exit-rad "$HEADING_ACTION_RECOVERY_EXIT_RAD" \
  --heading-action-recovery-gain "$HEADING_ACTION_RECOVERY_GAIN" \
  --heading-action-recovery-limit "$HEADING_ACTION_RECOVERY_LIMIT" \
  --cross-track-heading-limit "$CROSS_TRACK_HEADING_LIMIT" --heading-rate-limit "$HEADING_RATE_LIMIT" \
  --action-bias-mode "$ACTION_BIAS_MODE" --action-bias "$ACTION_BIAS" \
  --action-bias-ramp-seconds "$ACTION_BIAS_RAMP_SECONDS" \
  --ankle-roll-common-bias "$ANKLE_ROLL_COMMON_BIAS" \
  --left-hip-yaw-bias "$LEFT_HIP_YAW_BIAS" --right-hip-yaw-bias "$RIGHT_HIP_YAW_BIAS" \
  --yaw-action-gain "$YAW_ACTION_GAIN" \
  --turn-feedback-fade-seconds "$TURN_FEEDBACK_FADE_SECONDS" \
  --lateral-position-gain "$LATERAL_POSITION_GAIN" --lateral-velocity-gain "$LATERAL_VELOCITY_GAIN" \
  --recovery-enter-m "$RECOVERY_ENTER_M" --recovery-exit-m "$RECOVERY_EXIT_M" \
  --recovery-slew-rate-per-s "$RECOVERY_SLEW_RATE_PER_S" --phase-action-boost "$PHASE_ACTION_BOOST" \
  --action-ema-alpha "$ACTION_EMA_ALPHA" \
  --waist-tilt-action-multiplier "$WAIST_TILT_ACTION_MULTIPLIER" \
  --upper-scale "$UPPER_SCALE" --upper-start-seconds "$UPPER_START_SECONDS" \
  --upper-time-scale "$UPPER_TIME_SCALE" --upper-max-excursion-rad "$UPPER_MAX_EXCURSION_RAD" \
  --upper-max-velocity-radps "$UPPER_MAX_VELOCITY_RADPS" \
  --upper-fallback-tilt-rad "$UPPER_FALLBACK_TILT_RAD" \
  --upper-fallback-height-m "$UPPER_FALLBACK_HEIGHT_M" \
  --upper-fallback-heading-rad "$UPPER_FALLBACK_HEADING_RAD" \
  --upper-stop-mode "$UPPER_STOP_MODE" \
  --stationary-controller "$STATIONARY_CONTROLLER" --stop-controller "$STOP_CONTROLLER" \
  --stop-transition-seconds "$STOP_TRANSITION_SECONDS" \
  --curriculum-recovery-handoff-blend-seconds "$CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS" \
  --stop-intent-decelerate-seconds "$STOP_INTENT_DECELERATE_SECONDS" \
  --future-stop-preview-seconds "$FUTURE_STOP_PREVIEW_SECONDS" \
  --stop-brake-gain "$STOP_BRAKE_GAIN" --stop-brake-limit "$STOP_BRAKE_LIMIT" \
  --stop-brake-template-speed "$STOP_BRAKE_TEMPLATE_SPEED" \
  --stop-brake-template-floor "$STOP_BRAKE_TEMPLATE_FLOOR" \
  --event-hold-min-seconds "$EVENT_HOLD_MIN_SECONDS" --event-hold-speed "$EVENT_HOLD_SPEED" \
  --event-hold-tilt "$EVENT_HOLD_TILT" \
  --stop-emergency-speed-max "$STOP_EMERGENCY_SPEED_MAX" \
  --stop-emergency-min-seconds "$STOP_EMERGENCY_MIN_SECONDS" "${EXTRA_ARGS[@]}" \
  >"$LOG_ROOT/adapter.log" 2>&1 &
ADAPTER_PID=$!
sleep 1

cd "$SIM/bin"
printf '0\n' | ./start_sim.sh -s >"$LOG_ROOT/simulator.log" 2>&1 &
SIM_PID=$!
wait "$ADAPTER_PID"
tail -45 "$LOG_ROOT/adapter.log"
