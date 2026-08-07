#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
SIM="$ROOT/x2_rl_deploy_mujoco"
CASE_NAME="${CASE_NAME:?CASE_NAME is required}"
COMMAND_VX="${COMMAND_VX:-0.30}"
PHASE_OFFSET="${PHASE_OFFSET:-0.0}"
CONTROL_MODE="${CONTROL_MODE:-full}"
CLOCK_MODE="${CLOCK_MODE:-step}"
MIRROR_POLICY="${MIRROR_POLICY:-false}"
PD_PROFILE="${PD_PROFILE:-official_kp_ankle}"
DEFAULT_POSE_PROFILE="${DEFAULT_POSE_PROFILE:-stage208}"
PREPARE_SECONDS="${PREPARE_SECONDS:-0.2}"
STAND_SECONDS="${STAND_SECONDS:-2.0}"
MOVE_SECONDS="${MOVE_SECONDS:-4.0}"
STOP_SECONDS="${STOP_SECONDS:-8.0}"
STATE_PREDICTION_SECONDS="${STATE_PREDICTION_SECONDS:-0.0}"
STATE_QOS_DEPTH="${STATE_QOS_DEPTH:-10}"
HEADING_GAIN="${HEADING_GAIN:-0.0}"
HEADING_RECOVERY_ENTER_RAD="${HEADING_RECOVERY_ENTER_RAD:-}"
HEADING_RECOVERY_EXIT_RAD="${HEADING_RECOVERY_EXIT_RAD:-0.08}"
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
STOP_BRAKE_GAIN="${STOP_BRAKE_GAIN:-0.8}"
STOP_BRAKE_LIMIT="${STOP_BRAKE_LIMIT:-0.30}"
STOP_BRAKE_TEMPLATE_SPEED="${STOP_BRAKE_TEMPLATE_SPEED:-0.30}"
STOP_BRAKE_TEMPLATE_FLOOR="${STOP_BRAKE_TEMPLATE_FLOOR:-0.25}"
EVENT_HOLD_MIN_SECONDS="${EVENT_HOLD_MIN_SECONDS:-0.5}"
EVENT_HOLD_SPEED="${EVENT_HOLD_SPEED:-0.05}"
EVENT_HOLD_TILT="${EVENT_HOLD_TILT:-0.10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/results}"
MODEL_PATH="${MODEL_PATH:-/models/stage219_s2600_actor.onnx}"
STATIONARY_MODEL_PATH="${STATIONARY_MODEL_PATH:-/models/stand_backend_scratch_i150_actor.onnx}"
STATIONARY_WARMUP_SECONDS="${STATIONARY_WARMUP_SECONDS:-0.0}"
STATIONARY_BLEND="${STATIONARY_BLEND:-0.50}"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
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
[[ -n "$STATIONARY_MODEL_PATH" ]] && EXTRA_ARGS+=(--stationary-model "$STATIONARY_MODEL_PATH")
[[ "$MIRROR_POLICY" == "true" ]] && EXTRA_ARGS+=(--mirror-policy)

cd /repo
python3 tools/official_x2/stage208_official_mujoco_adapter.py \
  --model "$MODEL_PATH" --template /template.npz \
  --output "${OUTPUT_ROOT}/${CASE_NAME}.json" --vx "$COMMAND_VX" \
  --phase-offset "$PHASE_OFFSET" --clock-mode "$CLOCK_MODE" --control-mode "$CONTROL_MODE" \
  --pd-profile "$PD_PROFILE" --default-pose-profile "$DEFAULT_POSE_PROFILE" \
  --prepare-seconds "$PREPARE_SECONDS" --stand-seconds "$STAND_SECONDS" \
  --stationary-warmup-seconds "$STATIONARY_WARMUP_SECONDS" --stationary-blend "$STATIONARY_BLEND" \
  --move-seconds "$MOVE_SECONDS" --stop-seconds "$STOP_SECONDS" \
  --state-prediction-seconds "$STATE_PREDICTION_SECONDS" \
  --state-qos-depth "$STATE_QOS_DEPTH" \
  --heading-gain "$HEADING_GAIN" --cross-track-heading-gain "$CROSS_TRACK_HEADING_GAIN" \
  --heading-recovery-exit-rad "$HEADING_RECOVERY_EXIT_RAD" \
  --cross-track-heading-limit "$CROSS_TRACK_HEADING_LIMIT" --heading-rate-limit "$HEADING_RATE_LIMIT" \
  --action-bias-mode "$ACTION_BIAS_MODE" --action-bias "$ACTION_BIAS" \
  --action-bias-ramp-seconds "$ACTION_BIAS_RAMP_SECONDS" \
  --ankle-roll-common-bias "$ANKLE_ROLL_COMMON_BIAS" \
  --left-hip-yaw-bias "$LEFT_HIP_YAW_BIAS" --right-hip-yaw-bias "$RIGHT_HIP_YAW_BIAS" \
  --yaw-action-gain "$YAW_ACTION_GAIN" \
  --lateral-position-gain "$LATERAL_POSITION_GAIN" --lateral-velocity-gain "$LATERAL_VELOCITY_GAIN" \
  --recovery-enter-m "$RECOVERY_ENTER_M" --recovery-exit-m "$RECOVERY_EXIT_M" \
  --recovery-slew-rate-per-s "$RECOVERY_SLEW_RATE_PER_S" --phase-action-boost "$PHASE_ACTION_BOOST" \
  --action-ema-alpha "$ACTION_EMA_ALPHA" \
  --waist-tilt-action-multiplier "$WAIST_TILT_ACTION_MULTIPLIER" \
  --stationary-controller "$STATIONARY_CONTROLLER" --stop-controller "$STOP_CONTROLLER" \
  --stop-transition-seconds "$STOP_TRANSITION_SECONDS" \
  --stop-brake-gain "$STOP_BRAKE_GAIN" --stop-brake-limit "$STOP_BRAKE_LIMIT" \
  --stop-brake-template-speed "$STOP_BRAKE_TEMPLATE_SPEED" \
  --stop-brake-template-floor "$STOP_BRAKE_TEMPLATE_FLOOR" \
  --event-hold-min-seconds "$EVENT_HOLD_MIN_SECONDS" --event-hold-speed "$EVENT_HOLD_SPEED" \
  --event-hold-tilt "$EVENT_HOLD_TILT" "${EXTRA_ARGS[@]}" \
  >"$LOG_ROOT/adapter.log" 2>&1 &
ADAPTER_PID=$!
sleep 1

cd "$SIM/bin"
printf '0\n' | ./start_sim.sh -s >"$LOG_ROOT/simulator.log" 2>&1 &
SIM_PID=$!
wait "$ADAPTER_PID"
tail -45 "$LOG_ROOT/adapter.log"
