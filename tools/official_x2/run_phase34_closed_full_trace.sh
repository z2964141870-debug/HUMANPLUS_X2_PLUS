#!/usr/bin/env bash
set -euo pipefail

# The sole BASE Phase34 closed rollout.  The full-width mmap is intentionally
# external to the code repository (~624 MiB preallocated).
REPO=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
OFFICIAL=/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1
WORKSPACE="$OFFICIAL/worktree/x2_rl_deploy"
SIM="$WORKSPACE/x2_rl_deploy_mujoco"
RESULT="$OFFICIAL/results/phase34_full_stage250_trace"
MODEL_ROOT="$OFFICIAL/models"
TEMPLATE=/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz
DEFAULT_YAML="$OFFICIAL/vendor/aimdk-aarch64-a424add7-artifacts/extra/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml"
INNER="$REPO/tools/official_x2/run_official_gate_inner.sh"
SHIM="$OFFICIAL/cache/phase32_ldpreload_observer/libx2_phase32_mjtrace_glibc235.so"
CASE=phase34_stage250_closed_full_once_d230
DOMAIN=230
MAX_RECORDS=32768
MAX_TIME_S=15.0

mkdir -p "$RESULT"
[[ ! -e "$RESULT/$CASE.json" ]] || { echo "refusing to overwrite existing rollout" >&2; exit 2; }
[[ ! -e "$RESULT/$CASE.mmap" ]] || { echo "refusing to overwrite existing mmap" >&2; exit 2; }
[[ "$(sha256sum "$SIM/configuration/robot/lx2501_3_t2d5/model_info/scene.xml" | cut -d' ' -f1)" == \
   7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63 ]]
[[ "$(sha256sum "$MODEL_ROOT/stage219_s2600_actor.onnx" | cut -d' ' -f1)" == \
   b95bad3680658c7c25be50f236f070c80b7ff7ba8992355cec2ddfb1ee53c0f9 ]]
[[ "$(sha256sum "$MODEL_ROOT/stand_backend_scratch_i150_actor.onnx" | cut -d' ' -f1)" == \
   edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565 ]]
[[ "$(sha256sum "$TEMPLATE" | cut -d' ' -f1)" == \
   16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d ]]
[[ "$(sha256sum "$REPO/tools/official_x2/stage208_official_mujoco_adapter.py" | cut -d' ' -f1)" == \
   91a67ab6c583d48bbf09fea1d02de33f0b77961a404d8e67b7fab7764662d733 ]]
[[ "$(sha256sum "$SHIM" | cut -d' ' -f1)" == \
   ec4fe929a1ab1d01f71c1a7a939b512274f832dc15fad4ead725e396c9cfacd3 ]]

timeout 180s sg docker -c "docker run --rm --init --name x2-$CASE --network host --ipc host \
  -e ROS_DOMAIN_ID=$DOMAIN -e CASE_NAME=$CASE -e OUTPUT_ROOT=/results \
  -e LD_PRELOAD=/phase32/libx2_phase32_mjtrace.so \
  -e X2_MJ_SHIM_OUTPUT=/results/$CASE.mmap \
  -e X2_MJ_SHIM_MAX_RECORDS=$MAX_RECORDS -e X2_MJ_SHIM_MAX_TIME_S=$MAX_TIME_S \
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
