#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
SIM="$ROOT/x2_rl_deploy_mujoco"
OUTPUT_ROOT="${OUTPUT_ROOT:-/results}"
CAPTURE_NAME="${CAPTURE_NAME:-official_native_event_v2_20s}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-20}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is required}"
LOG_ROOT="$OUTPUT_ROOT/logs/$CAPTURE_NAME"
OUTPUT="$OUTPUT_ROOT/$CAPTURE_NAME.npz"

export ROS_DOMAIN_ID
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
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

mkdir -p "$SIM/bin/log/crash" "$SIM/bin/cfg/tmp" "$XDG_RUNTIME_DIR" "$LOG_ROOT" "$OUTPUT_ROOT"
chmod 700 "$XDG_RUNTIME_DIR"
cleanup() {
  kill "${RECORDER_PID:-}" "${CONTROLLER_PID:-}" "${SIM_PID:-}" "${XVFB_PID:-}" 2>/dev/null || true
  wait "${RECORDER_PID:-}" "${CONTROLLER_PID:-}" "${SIM_PID:-}" "${XVFB_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

Xvfb :99 -screen 0 1280x720x24 >"$LOG_ROOT/xvfb.log" 2>&1 &
XVFB_PID=$!
export DISPLAY=:99

cd "$ROOT"
ros2 run x2_rl_deploy_controller x2_rl_deploy_controller \
  >"$LOG_ROOT/controller.log" 2>&1 &
CONTROLLER_PID=$!

cd "$SIM/bin"
printf '0\n' | ./start_sim.sh -s >"$LOG_ROOT/simulator.log" 2>&1 &
SIM_PID=$!

# Require all four state and command groups plus root/IMU before mode switch.
required=(
  /aima/hal/joint/leg/state /aima/hal/joint/waist/state
  /aima/hal/joint/arm/state /aima/hal/joint/head/state
  /aima/hal/joint/leg/command /aima/hal/joint/waist/command
  /aima/hal/joint/arm/command /aima/hal/joint/head/command
  /aima/hal/odom/state /aima/hal/imu/torso/state /joy
)
ready=false
for _ in $(seq 1 200); do
  topics="$(ros2 topic list 2>/dev/null || true)"
  ready=true
  for topic in "${required[@]}"; do
    grep -qx "$topic" <<<"$topics" || ready=false
  done
  "$ready" && break
  kill -0 "$CONTROLLER_PID" "$SIM_PID" 2>/dev/null || break
  sleep 0.1
done
"$ready" || { echo "official source topics not ready" >&2; exit 3; }

# Follow the official README transition: JOINT_DEFAULT then RL_DEFAULT.
ros2 topic pub --once /joy sensor_msgs/msg/Joy '{buttons: [0, 0, 1, 0]}' \
  >"$LOG_ROOT/joy_joint_default.log" 2>&1
sleep 4
ros2 topic pub --once /joy sensor_msgs/msg/Joy '{buttons: [0, 0, 0, 1]}' \
  >"$LOG_ROOT/joy_rl_default.log" 2>&1
sleep 1

launch_identity="official README: start_sim.sh -s robot0 + ros2 run x2_rl_deploy_controller; Joy JOINT_DEFAULT 4s then RL_DEFAULT; isolated Docker/ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
python3 /repo/tools/official_x2/official_x2_rollout_recorder_v2.py \
  --duration "$CAPTURE_SECONDS" --output "$OUTPUT" \
  --ros-domain-id "$ROS_DOMAIN_ID" --control-mode RL_DEFAULT \
  --launch-identity "$launch_identity" \
  --official-onnx /workspace/x2_rl_deploy_controller/config/rl_model/kuailechongbai.onnx \
  --official-control-config /workspace/x2_rl_deploy_controller/config/motion_control.yaml \
  --official-scene /workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml \
  >"$LOG_ROOT/recorder.log" 2>&1 &
RECORDER_PID=$!
wait "$RECORDER_PID"
unset RECORDER_PID

ros2 topic pub --once /joy sensor_msgs/msg/Joy '{buttons: [0, 1, 0, 0]}' \
  >"$LOG_ROOT/joy_damping.log" 2>&1 || true

test -s "$OUTPUT"
test -s "${OUTPUT%.npz}.manifest.json"
tail -80 "$LOG_ROOT/recorder.log"
