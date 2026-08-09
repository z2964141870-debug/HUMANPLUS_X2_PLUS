#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
STAGE152="${ROOT}/scripts/run_dcpeft_stage152.sh"
CHECKPOINT="/home/humanplus/humanoid-GPT/A/sonic_release/last.pt"

ACTOR_LAYERS='[actor_module.decoders.g1_dyn.module.0,actor_module.decoders.g1_dyn.module.2,actor_module.decoders.g1_dyn.module.4,actor_module.decoders.g1_dyn.module.6,actor_module.decoders.g1_dyn.module.8,actor_module.decoders.g1_dyn.module.10,actor_module.decoders.g1_dyn.module.12]'
CRITIC_LAYERS='[critic_module.module.2,critic_module.module.4,critic_module.module.6,critic_module.module.8,critic_module.module.10]'

run_split() {
  local split="$1"
  local source_path="$2"
  local source_sha="$3"
  local keys_json="$4"
  local recorded_adapter="$5"
  # Both frozen artifacts contain exactly three keys. Loading one would invoke
  # MotionLib's random subset path and violate the immutable sampler contract.
  local motion_count=3

  PHASE46_SPLIT="${split}" \
  PHASE46_SOURCE_PATH="${source_path}" \
  PHASE46_SOURCE_SHA256="${source_sha}" \
  PHASE46_EXPECTED_KEYS_JSON="${keys_json}" \
  PHASE46_RECORDED_STATE_ADAPTER="${recorded_adapter}" \
  FAITHFUL_WBT29=true \
  CHECKPOINT="${CHECKPOINT}" \
  MOTION_FILE="${source_path}" \
  RUN_NAME="x2_faithful_live_zero_phase46_${split}" \
  RUN_KIND="contract_zero" \
  SEED=0 \
  NUM_ENVS=1 \
  STEPS_PER_ENV=2 \
  ITERS=0 \
  MOTIONS="${motion_count}" \
  ADAPTIVE_SAMPLING_ENABLED=false \
  START_FROM_FIRST_FRAME=true \
  FREEZE_FRAME_AUG=true \
  CLEAN_EVAL=true \
  CLEAN_COMMAND_INIT=true \
  ENCODER_SAMPLE_PROBS='{g1:1.0,teleop:0.0,smpl:0.0}' \
  X2_COLLISION_PROFILE=sole12 \
  X2_PD_PROFILE=natural_frequency \
  X2_ACTION_SCALE_PROFILE=torque_normalized \
  ACTION_SCALE_MULT=1.0 \
  SIM_DT=0.005 \
  DECIMATION=4 \
  ACTUATOR_RESPONSE_MODE=off \
  RIGID_BODY_MASS_RANGE='[0.8,1.5]' \
  PPO_EPOCHS=5 \
  NUM_MINI_BATCHES=4 \
  ACTOR_LORA_PREFIXES="${ACTOR_LAYERS}" \
  CRITIC_LORA_PREFIXES="${CRITIC_LAYERS}" \
  ACTOR_INPUT_MASK_ENABLED=true \
  ACTOR_INPUT_MASK_LAYERS='[actor_module.decoders.g1_dyn.module.0]' \
  ACTOR_INPUT_MASK_DECODER=g1_dyn \
  ACTOR_INPUT_MASK_FEATURES='[proprioception]' \
  ACTOR_OUTPUT_MASK_ENABLED=false \
  TRAIN_BOUNDARY_KEYS='[]' \
  PRESERVE_LORA_CHECKPOINT=false \
  STACK_LORA_CHECKPOINT_RESIDUAL=false \
  SEMANTIC_COPY_STRICT=true \
  SAVE_FREQUENCY=-1 \
  SAVE_LAST_FREQUENCY=-1 \
  TIMEOUT_SECONDS=900 \
  bash "${STAGE152}"
}

case "${1:-both}" in
  train)
    run_split train \
      "${ROOT}/artifacts/retarget/x2_phase45_kinematic_bronze_train.pkl" \
      "06b759e7926bd841537cf515d7cc46793d47503a9fab5da95c649a2f0b6ec4a6" \
      '["AMASS-STAND-001","AMASS-UPPER-001","PHUMA-LUNGE-R-001"]' false
    ;;
  held_out)
    run_split held_out \
      "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/gold_dynamic_native_seed_v1/held_out/official_native_dance_held_out.pkl" \
      "45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc" \
      '["official_native_dance_held_out_000","official_native_dance_held_out_001","official_native_dance_held_out_002"]' true
    ;;
  both)
    "$0" train
    "$0" held_out
    ;;
  *) echo "usage: $0 [train|held_out|both]" >&2; exit 2 ;;
esac
