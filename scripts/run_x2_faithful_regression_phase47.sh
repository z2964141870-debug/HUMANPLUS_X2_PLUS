#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
STAGE152="${ROOT}/scripts/run_dcpeft_stage152.sh"
SOURCE_CHECKPOINT="/home/humanplus/humanoid-GPT/A/sonic_release/last.pt"
FINAL_CHECKPOINT="${ROOT}/logs/x2_faithful_exact_s7_one_update_phase47/last.pt"
SOURCE_SHA="e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
FINAL_SHA="$(sha256sum "${FINAL_CHECKPOINT}" | awk '{print $1}')"
BRONZE="${ROOT}/artifacts/retarget/x2_phase45_kinematic_bronze_train.pkl"
GOLD="/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/gold_dynamic_native_seed_v1/held_out/official_native_dance_held_out.pkl"
ACTOR_LAYERS='[actor_module.decoders.g1_dyn.module.0,actor_module.decoders.g1_dyn.module.2,actor_module.decoders.g1_dyn.module.4,actor_module.decoders.g1_dyn.module.6,actor_module.decoders.g1_dyn.module.8,actor_module.decoders.g1_dyn.module.10,actor_module.decoders.g1_dyn.module.12]'
CRITIC_LAYERS='[critic_module.module.2,critic_module.module.4,critic_module.module.6,critic_module.module.8,critic_module.module.10]'

run_eval() {
  local role="$1" split="$2" motion="$3" motion_sha="$4" keys="$5" adapter="$6" horizon="$7" checkpoint="$8" checkpoint_sha="$9" preserve="${10}"
  local label="${role}_${split}"
  PHASE47_EVAL_LABEL="${label}" \
  PHASE47_EVAL_SPLIT="${split}" \
  PHASE47_EVAL_SOURCE_PATH="${motion}" \
  PHASE47_EVAL_SOURCE_SHA256="${motion_sha}" \
  PHASE47_EVAL_KEYS_JSON="${keys}" \
  PHASE47_EVAL_RECORDED_ADAPTER="${adapter}" \
  PHASE47_EVAL_HORIZON="${horizon}" \
  PHASE47_EVAL_CHECKPOINT_SHA256="${checkpoint_sha}" \
  FAITHFUL_WBT29=true \
  FAITHFUL_WBT29_TRAINER_TARGET=x2_faithful_one_update_phase47.Phase47BoundedRegressionTrainer \
  CHECKPOINT="${checkpoint}" MOTION_FILE="${motion}" \
  RUN_NAME="x2_faithful_phase47_regression_${label}" RUN_KIND=bounded_regression \
  SEED=0 NUM_ENVS=12 STEPS_PER_ENV=2 ITERS=1 MOTIONS=3 \
  ADAPTIVE_SAMPLING_ENABLED=false START_FROM_FIRST_FRAME=true FREEZE_FRAME_AUG=true \
  CLEAN_EVAL=true CLEAN_COMMAND_INIT=true \
  ENCODER_SAMPLE_PROBS='{g1:1.0,teleop:0.0,smpl:0.0}' \
  X2_COLLISION_PROFILE=sole12 X2_PD_PROFILE=natural_frequency \
  X2_ACTION_SCALE_PROFILE=torque_normalized ACTION_SCALE_MULT=1.0 \
  SIM_DT=0.005 DECIMATION=4 ACTUATOR_RESPONSE_MODE=off \
  PPO_EPOCHS=5 NUM_MINI_BATCHES=4 GRADIENT_ACCUMULATION_STEPS=1 \
  DETERMINISTIC_ROLLOUT=true \
  ACTOR_LORA_PREFIXES="${ACTOR_LAYERS}" CRITIC_LORA_PREFIXES="${CRITIC_LAYERS}" \
  ACTOR_INPUT_MASK_ENABLED=true \
  ACTOR_INPUT_MASK_LAYERS='[actor_module.decoders.g1_dyn.module.0]' \
  ACTOR_INPUT_MASK_DECODER=g1_dyn ACTOR_INPUT_MASK_FEATURES='[proprioception]' \
  TRAIN_BOUNDARY_KEYS='[]' PRESERVE_LORA_CHECKPOINT="${preserve}" \
  STACK_LORA_CHECKPOINT_RESIDUAL=false SEMANTIC_COPY_STRICT=true \
  SAVE_FREQUENCY=-1 SAVE_LAST_FREQUENCY=-1 TRAINING_DIAGNOSTICS_ENABLED=false \
  TIMEOUT_SECONDS=1200 bash "${STAGE152}"
}

BRONZE_KEYS='["AMASS-STAND-001","AMASS-UPPER-001","PHUMA-LUNGE-R-001"]'
GOLD_KEYS='["official_native_dance_held_out_000","official_native_dance_held_out_001","official_native_dance_held_out_002"]'

case "${1:-all}" in
  pre_train|all)
    run_eval pre train "${BRONZE}" 06b759e7926bd841537cf515d7cc46793d47503a9fab5da95c649a2f0b6ec4a6 "${BRONZE_KEYS}" false 175 "${SOURCE_CHECKPOINT}" "${SOURCE_SHA}" false
    [[ "${1:-all}" != all ]] && exit 0
    ;;
esac
case "${1:-all}" in
  post_train|all)
    run_eval post train "${BRONZE}" 06b759e7926bd841537cf515d7cc46793d47503a9fab5da95c649a2f0b6ec4a6 "${BRONZE_KEYS}" false 175 "${FINAL_CHECKPOINT}" "${FINAL_SHA}" true
    [[ "${1:-all}" != all ]] && exit 0
    ;;
esac
case "${1:-all}" in
  pre_gold|all)
    run_eval pre held_out "${GOLD}" 45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc "${GOLD_KEYS}" true 400 "${SOURCE_CHECKPOINT}" "${SOURCE_SHA}" false
    [[ "${1:-all}" != all ]] && exit 0
    ;;
esac
case "${1:-all}" in
  post_gold|all)
    run_eval post held_out "${GOLD}" 45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc "${GOLD_KEYS}" true 400 "${FINAL_CHECKPOINT}" "${FINAL_SHA}" true
    [[ "${1:-all}" != all ]] && exit 0
    ;;
  *) echo "usage: $0 [all|pre_train|post_train|pre_gold|post_gold]" >&2; exit 2 ;;
esac
