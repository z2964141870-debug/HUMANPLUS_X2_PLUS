#!/usr/bin/env bash
# Run on SoC1 only. Diagnostic computation; never use this for motor takeover.
set -euo pipefail
root=/agibot/data/home/agi/gear-sonic
binary="$root/ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"
model="$root/policies/x2_sonic_s1ft16000_g1.onnx"
log_root=${1:?Supply a new isolated diagnostic log directory}
test -d "$log_root"
test ! -e "$log_root/csv"
printf '%s  %s\n' \
  0b02df145874c3e4c6cb05ea4778f09b1bae372212c5bf3754f561b64791864a "$binary" \
  7dfdec7cf1b1ac67635c3cd8f541c27b50d917df2e3381a6bb086b605e1959a0 "$model" \
  | sha256sum --check --strict
set +u
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH="$root/../aimdk_msgs_pc2:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$root/onnxruntime/lib:$root/../aimdk_msgs_pc2/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root/../aimdk_msgs_pc2/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
set -u
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/data/home/agi/.aima/env/ros_dds_configuration.xml
unset X2_PLANT_YAML
export X2_PLANT=vendor_20260823 X2_ANCHOR_ORI_MODE=body X2_WHOLE_BODY_TELEOP=0
export X2_REPO_ROOT="$root"
cd "$root"
args=(--no-publish --model "$model" --input-type motion_file
  --autostart-after 2 --max-duration 20 --return-seconds 0
  --hold-for-mc-timeout-s 0 --mc-mode-poll-s 0 --log-dir "$log_root/csv"
  --action-clip 20 --max-target-dev 0.30 --max-target-dev-leg 0.70
  --max-target-dev-waist 0.45 --max-target-dev-arm 2.00 --max-target-dev-head 0.50
  --ramp-seconds 2 --tilt-cos -0.3 --target-lpf-hz 8
  --target-lpf-hz-leg 16 --target-lpf-hz-waist 0 --target-lpf-hz-arm 0
  --kp-scale-ankle-pitch 1 --kp-scale-ankle-roll 1 --kp-scale-knee 1
  --kp-scale-waist-yaw 1 --kp-scale-waist-pr 2
  --kd-scale-ankle-pitch 1 --kd-scale-ankle-roll 1 --kd-scale-knee 1
  --kd-scale-hip 1 --kd-scale-waist-yaw 1 --kd-scale-waist-pr 5.51)
printf 'OBSERVE_ONLY_COMMAND: timeout -s TERM 40 %q ' "$binary"
printf '%q ' "${args[@]}"
printf '\n'
exec timeout -s TERM 40 "$binary" "${args[@]}"
