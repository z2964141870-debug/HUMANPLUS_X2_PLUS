#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
STAGE152="${ROOT}/scripts/run_dcpeft_stage152.sh"
SEGMENT="${1:?segment 1..5 required}"
CHECKPOINT="${2:?checkpoint required}"
EXPECTED_SHA="${3:?checkpoint sha required}"
CHECKPOINT="$(realpath "${CHECKPOINT}")"
RUN_NAME="x2_faithful_exact_s7_phase49_update$(printf '%02d' "${SEGMENT}")"
LOG_DIR="${ROOT}/logs/${RUN_NAME}"
TRAIN="${ROOT}/artifacts/retarget/x2_phase45_kinematic_bronze_train.pkl"
ACTOR_LAYERS='[actor_module.decoders.g1_dyn.module.0,actor_module.decoders.g1_dyn.module.2,actor_module.decoders.g1_dyn.module.4,actor_module.decoders.g1_dyn.module.6,actor_module.decoders.g1_dyn.module.8,actor_module.decoders.g1_dyn.module.10,actor_module.decoders.g1_dyn.module.12]'
CRITIC_LAYERS='[critic_module.module.2,critic_module.module.4,critic_module.module.6,critic_module.module.8,critic_module.module.10]'

[[ "${SEGMENT}" =~ ^[1-5]$ ]] || { echo "segment must be 1..5" >&2; exit 2; }
[[ ! -e "${LOG_DIR}" ]] || { echo "existing Phase49 run dir: ${LOG_DIR}" >&2; exit 2; }
if [[ "${SEGMENT}" == 1 ]]; then
  RESUME=false; PRESERVE=false; PRETRAIN="${LOG_DIR}/source_B0.pt"
else
  RESUME=true; PRESERVE=true; PRETRAIN=null
fi

PHASE49_SEGMENT="${SEGMENT}" \
PHASE49_EXPECTED_CHECKPOINT_SHA256="${EXPECTED_SHA}" \
FAITHFUL_WBT29=true \
FAITHFUL_WBT29_TRAINER_TARGET=x2_faithful_five_update_phase49.Phase49GatedSegmentTrainer \
CHECKPOINT="${CHECKPOINT}" MOTION_FILE="${TRAIN}" RUN_NAME="${RUN_NAME}" RUN_KIND=five_update_gated_segment \
SEED=0 NUM_ENVS=64 STEPS_PER_ENV=24 ITERS=5 MOTIONS=3 \
RESUME="${RESUME}" RESUME_KEEP_EXPERIMENT_DIR=true \
ADAPTIVE_SAMPLING_ENABLED=false START_FROM_FIRST_FRAME=true FREEZE_FRAME_AUG=true \
CLEAN_EVAL=true CLEAN_COMMAND_INIT=true ENCODER_SAMPLE_PROBS='{g1:1.0,teleop:0.0,smpl:0.0}' \
X2_COLLISION_PROFILE=sole12 X2_PD_PROFILE=natural_frequency X2_ACTION_SCALE_PROFILE=torque_normalized \
ACTION_SCALE_MULT=1.0 SIM_DT=0.005 DECIMATION=4 ACTUATOR_RESPONSE_MODE=off RIGID_BODY_MASS_RANGE='[0.8,1.5]' \
PPO_EPOCHS=5 NUM_MINI_BATCHES=4 GRADIENT_ACCUMULATION_STEPS=1 \
PPO_SHUFFLE_ENABLED=true PPO_SHUFFLE_EVERY_EPOCH=true DETERMINISTIC_ROLLOUT=false \
ACTOR_LORA_PREFIXES="${ACTOR_LAYERS}" CRITIC_LORA_PREFIXES="${CRITIC_LAYERS}" \
ACTOR_INPUT_MASK_ENABLED=true ACTOR_INPUT_MASK_LAYERS='[actor_module.decoders.g1_dyn.module.0]' \
ACTOR_INPUT_MASK_DECODER=g1_dyn ACTOR_INPUT_MASK_FEATURES='[proprioception]' \
ACTOR_OUTPUT_MASK_ENABLED=false TRAIN_BOUNDARY_KEYS='[]' \
PRESERVE_LORA_CHECKPOINT="${PRESERVE}" STACK_LORA_CHECKPOINT_RESIDUAL=false SEMANTIC_COPY_STRICT=true \
TRAINING_DIAGNOSTICS_ENABLED=true PRETRAIN_EXPORT_PATH="${PRETRAIN}" \
SAVE_FREQUENCY=999999 SAVE_LAST_FREQUENCY=1 TIMEOUT_SECONDS=3600 bash "${STAGE152}"
