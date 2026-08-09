#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
STAGE152="${ROOT}/scripts/run_dcpeft_stage152.sh"
SEGMENT="${1:?segment required}"
ROLE="${2:?train or held_out required}"
CHECKPOINT="${3:?checkpoint required}"
CHECKPOINT_SHA="${4:?checkpoint sha required}"
CHECKPOINT="$(realpath "${CHECKPOINT}")"
BRONZE="${ROOT}/artifacts/retarget/x2_phase45_kinematic_bronze_train.pkl"
GOLD="/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/gold_dynamic_native_seed_v1/held_out/official_native_dance_held_out.pkl"
ACTOR_LAYERS='[actor_module.decoders.g1_dyn.module.0,actor_module.decoders.g1_dyn.module.2,actor_module.decoders.g1_dyn.module.4,actor_module.decoders.g1_dyn.module.6,actor_module.decoders.g1_dyn.module.8,actor_module.decoders.g1_dyn.module.10,actor_module.decoders.g1_dyn.module.12]'
CRITIC_LAYERS='[critic_module.module.2,critic_module.module.4,critic_module.module.6,critic_module.module.8,critic_module.module.10]'
LABEL="phase49_u$(printf '%02d' "${SEGMENT}")_${ROLE}"

if [[ "${ROLE}" == train ]]; then
  MOTION="${BRONZE}"; MOTION_SHA=06b759e7926bd841537cf515d7cc46793d47503a9fab5da95c649a2f0b6ec4a6
  KEYS='["AMASS-STAND-001","AMASS-UPPER-001","PHUMA-LUNGE-R-001"]'; ADAPTER=false; HORIZON=175; SPLIT=train
elif [[ "${ROLE}" == held_out ]]; then
  MOTION="${GOLD}"; MOTION_SHA=45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc
  KEYS='["official_native_dance_held_out_000","official_native_dance_held_out_001","official_native_dance_held_out_002"]'; ADAPTER=true; HORIZON=400; SPLIT=held_out
else
  echo "role must be train or held_out" >&2; exit 2
fi

PHASE47_EVAL_LABEL="${LABEL}" PHASE47_EVAL_SPLIT="${SPLIT}" \
PHASE47_EVAL_SOURCE_PATH="${MOTION}" PHASE47_EVAL_SOURCE_SHA256="${MOTION_SHA}" \
PHASE47_EVAL_KEYS_JSON="${KEYS}" PHASE47_EVAL_RECORDED_ADAPTER="${ADAPTER}" \
PHASE47_EVAL_HORIZON="${HORIZON}" PHASE47_EVAL_CHECKPOINT_SHA256="${CHECKPOINT_SHA}" \
FAITHFUL_WBT29=true FAITHFUL_WBT29_TRAINER_TARGET=x2_faithful_one_update_phase47.Phase47BoundedRegressionTrainer \
CHECKPOINT="${CHECKPOINT}" MOTION_FILE="${MOTION}" \
RUN_NAME="x2_faithful_phase49_regression_${LABEL}" RUN_KIND=bounded_regression \
SEED=0 NUM_ENVS=12 STEPS_PER_ENV=2 ITERS=1 MOTIONS=3 \
ADAPTIVE_SAMPLING_ENABLED=false START_FROM_FIRST_FRAME=true FREEZE_FRAME_AUG=true \
CLEAN_EVAL=true CLEAN_COMMAND_INIT=true ENCODER_SAMPLE_PROBS='{g1:1.0,teleop:0.0,smpl:0.0}' \
X2_COLLISION_PROFILE=sole12 X2_PD_PROFILE=natural_frequency X2_ACTION_SCALE_PROFILE=torque_normalized \
ACTION_SCALE_MULT=1.0 SIM_DT=0.005 DECIMATION=4 ACTUATOR_RESPONSE_MODE=off \
PPO_EPOCHS=5 NUM_MINI_BATCHES=4 GRADIENT_ACCUMULATION_STEPS=1 DETERMINISTIC_ROLLOUT=true \
ACTOR_LORA_PREFIXES="${ACTOR_LAYERS}" CRITIC_LORA_PREFIXES="${CRITIC_LAYERS}" \
ACTOR_INPUT_MASK_ENABLED=true ACTOR_INPUT_MASK_LAYERS='[actor_module.decoders.g1_dyn.module.0]' \
ACTOR_INPUT_MASK_DECODER=g1_dyn ACTOR_INPUT_MASK_FEATURES='[proprioception]' \
TRAIN_BOUNDARY_KEYS='[]' PRESERVE_LORA_CHECKPOINT=true STACK_LORA_CHECKPOINT_RESIDUAL=false \
SEMANTIC_COPY_STRICT=true SAVE_FREQUENCY=-1 SAVE_LAST_FREQUENCY=-1 \
TRAINING_DIAGNOSTICS_ENABLED=false TIMEOUT_SECONDS=1200 bash "${STAGE152}"
