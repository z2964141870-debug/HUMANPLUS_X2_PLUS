#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
SIM="$ROOT/x2_rl_deploy_mujoco"
OUTPUT_ROOT="${OUTPUT_ROOT:-/results}"
CAPTURE_NAME="${CAPTURE_NAME:-official_native_event_v2_activation_prefix_phase20}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:?ROS_DOMAIN_ID is required}"
RL_SECONDS="${RL_SECONDS:-20}"
LOG_ROOT="$OUTPUT_ROOT/logs/$CAPTURE_NAME"
OUTPUT="$OUTPUT_ROOT/$CAPTURE_NAME.npz"
SUB_READY=/tmp/x2_recorder_subscription_ready
SNAP_READY=/tmp/x2_recorder_snapshot_ready
STOP=/tmp/x2_recorder_stop
READINESS="$OUTPUT_ROOT/$CAPTURE_NAME.readiness.json"

export ROS_DOMAIN_ID RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="/repo/tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/install/aimdk_msgs/lib:$SIM/bin:$SIM/lib:/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/onnxruntime/lib"
export SIM_ROBOT_PATH="$SIM/configuration/robot/lx2501_3_t2d5"
export SIM_RESOURCE_MODEL_PATH="$SIM/resource/model"
export XDG_RUNTIME_DIR=/tmp/runtime-root
export AGIBOT_ENABLE_HDS_COMPONENT=false AGIBOT_ENABLE_EVENT_COMPONENT=false
export AGIBOT_ENABLE_AUDIT_COMPONENT=false AGIBOT_ENABLE_MONITOR=true
export EM_APP_NAME=sim AIMRT_PLUGIN_SHM_DEFAULT_WAIT_TIME_US=300

mkdir -p "$SIM/bin/log/crash" "$SIM/bin/cfg/tmp" "$XDG_RUNTIME_DIR" "$LOG_ROOT" "$OUTPUT_ROOT"
chmod 700 "$XDG_RUNTIME_DIR"
rm -f "$SUB_READY" "$SNAP_READY" "$STOP"
cleanup() {
  kill "${JOY_PID:-}" "${RECORDER_PID:-}" "${CONTROLLER_PID:-}" "${SIM_PID:-}" "${XVFB_PID:-}" 2>/dev/null || true
  wait "${JOY_PID:-}" "${RECORDER_PID:-}" "${CONTROLLER_PID:-}" "${SIM_PID:-}" "${XVFB_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

Xvfb :99 -screen 0 1280x720x24 >"$LOG_ROOT/xvfb.log" 2>&1 & XVFB_PID=$!
export DISPLAY=:99
cd "$ROOT"
ros2 run x2_rl_deploy_controller x2_rl_deploy_controller >"$LOG_ROOT/controller.log" 2>&1 & CONTROLLER_PID=$!
cd "$SIM/bin"
printf '0\n' | ./start_sim.sh -s >"$LOG_ROOT/simulator.log" 2>&1 & SIM_PID=$!

required=(
  /aima/hal/joint/leg/state /aima/hal/joint/waist/state /aima/hal/joint/arm/state /aima/hal/joint/head/state
  /aima/hal/joint/leg/command /aima/hal/joint/waist/command /aima/hal/joint/arm/command /aima/hal/joint/head/command
  /aima/hal/odom/state /aima/hal/imu/torso/state /joy
)
topic_ready=false
for _ in $(seq 1 200); do
  topics="$(ros2 topic list 2>/dev/null || true)"; topic_ready=true
  for topic in "${required[@]}"; do grep -qx "$topic" <<<"$topics" || topic_ready=false; done
  "$topic_ready" && break
  kill -0 "$CONTROLLER_PID" "$SIM_PID" 2>/dev/null || break
  sleep 0.1
done
"$topic_ready" || { echo "official topics not available" >&2; exit 3; }

launch_identity="Phase20 activation: recorder subscription-ready; JOINT activates telemetry; first complete snapshot + >=4s JOINT; RL ${RL_SECONDS}s; domain=$ROS_DOMAIN_ID"
python3 /repo/tools/official_x2/official_x2_rollout_recorder_v2.py \
  --duration 60 --output "$OUTPUT" --subscription-ready-file "$SUB_READY" \
  --ready-file "$SNAP_READY" --stop-file "$STOP" --readiness-diagnostics "$READINESS" \
  --ros-domain-id "$ROS_DOMAIN_ID" --control-mode 'JOINT_DEFAULT->RL_DEFAULT' \
  --launch-identity "$launch_identity" \
  --official-onnx /workspace/x2_rl_deploy_controller/config/rl_model/kuailechongbai.onnx \
  --official-control-config /workspace/x2_rl_deploy_controller/config/motion_control.yaml \
  --official-scene /workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml \
  >"$LOG_ROOT/recorder.log" 2>&1 & RECORDER_PID=$!

for _ in $(seq 1 100); do test -s "$SUB_READY" && break; kill -0 "$RECORDER_PID" || exit 4; sleep 0.05; done
test -s "$SUB_READY" || { echo "recorder subscription-ready missing" >&2; exit 4; }

ros2 topic pub --once /joy sensor_msgs/msg/Joy '{buttons: [0, 0, 1, 0]}' >"$LOG_ROOT/joy_joint_default.log" 2>&1 & JOY_PID=$!
for _ in $(seq 1 100); do grep -q 'Switch to JOINT_DEFAULT' "$LOG_ROOT/controller.log" && break; sleep 0.05; done
grep -q 'Switch to JOINT_DEFAULT' "$LOG_ROOT/controller.log" || { echo "JOINT_DEFAULT not acknowledged" >&2; exit 5; }
wait "$JOY_PID"; unset JOY_PID

# Hard state gate: JOINT is already active, but the four-second prefix clock
# starts only when an actual complete snapshot has been stored.
for _ in $(seq 1 300); do test -s "$SNAP_READY" && break; kill -0 "$RECORDER_PID" || exit 6; sleep 0.05; done
test -s "$SNAP_READY" || { echo "complete snapshot not ready after JOINT" >&2; cat "$READINESS" >&2; exit 6; }
sleep 4

ros2 topic pub --once /joy sensor_msgs/msg/Joy '{buttons: [0, 0, 0, 1]}' >"$LOG_ROOT/joy_rl_default.log" 2>&1 & JOY_PID=$!
for _ in $(seq 1 100); do grep -q 'Switch to RL_DEFAULT' "$LOG_ROOT/controller.log" && break; sleep 0.05; done
grep -q 'Switch to RL_DEFAULT' "$LOG_ROOT/controller.log" || { echo "RL_DEFAULT not acknowledged" >&2; exit 7; }
wait "$JOY_PID"; unset JOY_PID
sleep "$RL_SECONDS"
touch "$STOP"
wait "$RECORDER_PID"; unset RECORDER_PID
test -s "$OUTPUT"; test -s "${OUTPUT%.npz}.manifest.json"
tail -80 "$LOG_ROOT/recorder.log"
