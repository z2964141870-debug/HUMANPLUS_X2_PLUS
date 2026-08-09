#!/usr/bin/env bash
set -euo pipefail

# Exactly one isolated closed AimDK capture for BASE Phase32.  This leaves the
# vendor binary, model, controller, and Stage250 contract unchanged; only the
# default-off observer is mounted and preloaded.
REPO=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
OFFICIAL=/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1
WORKSPACE="$OFFICIAL/worktree/x2_rl_deploy"
SIM="$WORKSPACE/x2_rl_deploy_mujoco"
RESULT="$OFFICIAL/results/phase32_closed_prefix"
MODEL_ROOT="$OFFICIAL/models"
TEMPLATE=/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz
DEFAULT_YAML="$OFFICIAL/vendor/aimdk-aarch64-a424add7-artifacts/extra/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml"
INNER="$REPO/tools/official_x2/run_official_gate_inner.sh"
SHIM="$OFFICIAL/cache/phase32_ldpreload_observer/libx2_phase32_mjtrace_glibc235.so"
CASE=phase32_stage250_closed_prefix_once
DOMAIN=232

mkdir -p "$RESULT"
[[ ! -e "$RESULT/$CASE.json" ]] || { echo "refusing to overwrite existing capture" >&2; exit 2; }
[[ ! -e "$RESULT/$CASE.mmap" ]] || { echo "refusing to overwrite existing mmap" >&2; exit 2; }
[[ "$(sha256sum "$SIM/configuration/robot/lx2501_3_t2d5/model_info/scene.xml" | cut -d' ' -f1)" == \
   7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63 ]]
[[ "$(sha256sum "$SHIM" | cut -d' ' -f1)" == \
   ec4fe929a1ab1d01f71c1a7a939b512274f832dc15fad4ead725e396c9cfacd3 ]]

timeout 180s sg docker -c "docker run --rm --init --name x2-$CASE --network host --ipc host \
  -e ROS_DOMAIN_ID=$DOMAIN -e CASE_NAME=$CASE -e OUTPUT_ROOT=/results \
  -e LD_PRELOAD=/phase32/libx2_phase32_mjtrace.so \
  -e X2_MJ_SHIM_OUTPUT=/results/$CASE.mmap \
  -e X2_MJ_SHIM_MAX_RECORDS=4096 -e X2_MJ_SHIM_MAX_TIME_S=0.3 \
  -e MODEL_PATH=/models/stage219_s2600_actor.onnx \
  -e COMMAND_VX=0.30 -e STATE_QOS_DEPTH=1 -e STATE_PREDICTION_SECONDS=0 \
  -e MOVE_SECONDS=4.0 -e STOP_SECONDS=8.0 -e STOP_CONTROLLER=policy \
  -e HEADING_GAIN=0.50 -e HEADING_RATE_LIMIT=0.10 \
  -e ACTION_BIAS_MODE=lateral_recovery_supervisor -e ACTION_BIAS=0.50 \
  -e ANKLE_ROLL_COMMON_BIAS=0.20 -e RECOVERY_ENTER_M=0.12 \
  -e RECOVERY_EXIT_M=0.04 -e RECOVERY_SLEW_RATE_PER_S=1.0 \
  -v $INNER:/run_official_gate_inner.sh:ro \
  -v $WORKSPACE:/workspace -v $REPO:/repo:ro -v $MODEL_ROOT:/models:ro \
  -v $TEMPLATE:/template.npz:ro -v $RESULT:/results \
  -v $DEFAULT_YAML:/workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml:ro \
  -v $SHIM:/phase32/libx2_phase32_mjtrace.so:ro \
  x2-aimdk-humble:1.0 bash -lc 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; bash /run_official_gate_inner.sh'"

[[ -s "$RESULT/$CASE.json" ]]
[[ -s "$RESULT/$CASE.mmap" ]]
echo "$RESULT/$CASE.json"
echo "$RESULT/$CASE.mmap"
