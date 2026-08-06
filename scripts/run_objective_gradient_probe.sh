#!/usr/bin/env bash
set -euo pipefail

# Non-mutating actor-objective gradient probe.
# Usage: bash scripts/run_objective_gradient_probe.sh {E1|N1} {INIT|I25} SEED

PROJECT_ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
SOURCE_ROOT=/home/humanplus/x2_teleop_final/x2_sonic
BASE_LAUNCHER="${PROJECT_ROOT}/scripts/run_dcpeft_stage152.sh"
DUAL_INIT="${PROJECT_ROOT}/checkpoints/stage152_B_dual_equal_split_init.pt"
MOTION_FILE="${SOURCE_ROOT}/motion_lib_x2/stage72_official_true_forward4_v1"

ARM=$(printf '%s' "${1:?arm required: E1|N1}" | tr '[:lower:]' '[:upper:]')
STAGE=$(printf '%s' "${2:?stage required: INIT|I25}" | tr '[:lower:]' '[:upper:]')
SEED="${3:?seed required}"
NUM_ENVS="${NUM_ENVS:-32}"
STEPS_PER_ENV="${STEPS_PER_ENV:-24}"
RUN_ID="${RUN_ID:-v1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"

case "${ARM}" in
  E1)
    CONTRACT="${PROJECT_ROOT}/configs/reward_contract_dual_v1.json"
    I25_CHECKPOINT="${PROJECT_ROOT}/logs/dcpeft_e1_from_stage152b_i25_seed0_v1/model_step_000025.pt"
    ;;
  N1)
    CONTRACT="${PROJECT_ROOT}/configs/reward_contract_bad_materialized_v1.json"
    I25_CHECKPOINT="${PROJECT_ROOT}/logs/dcpeft_n1_from_stage152b_i25_seed0_v1/model_step_000025.pt"
    ;;
  *)
    echo "usage: $0 {E1|N1} {INIT|I25} SEED" >&2
    exit 2
    ;;
esac

case "${STAGE}" in
  INIT) CHECKPOINT="${DUAL_INIT}" ;;
  I25) CHECKPOINT="${I25_CHECKPOINT}" ;;
  *)
    echo "usage: $0 {E1|N1} {INIT|I25} SEED" >&2
    exit 2
    ;;
esac

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED must be non-negative" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${CONTRACT}" ]] || { echo "missing contract: ${CONTRACT}" >&2; exit 2; }
[[ -d "${MOTION_FILE}" ]] || { echo "missing motion set: ${MOTION_FILE}" >&2; exit 2; }

RUN_NAME="objective_gradient_${ARM,,}_${STAGE,,}_seed${SEED}_${RUN_ID}"
RUN_DIR="${PROJECT_ROOT}/logs/${RUN_NAME}"
OUTPUT="${PROJECT_ROOT}/results/${RUN_NAME}.jsonl"
if [[ -e "${RUN_DIR}" || -e "${OUTPUT}" ]]; then
  echo "refusing to reuse probe output: ${RUN_NAME}" >&2
  exit 3
fi

echo "objective_gradient arm=${ARM} stage=${STAGE} seed=${SEED} checkpoint=${CHECKPOINT}"

exec env \
  RUN_NAME="${RUN_NAME}" \
  RUN_KIND=dcpeft_objective_gradient_probe \
  CHECKPOINT="${CHECKPOINT}" \
  MOTION_FILE="${MOTION_FILE}" \
  EXP_CONFIG=manager/universal_token/all_modes/sonic_x2_phuma_response_context \
  SEED="${SEED}" \
  NUM_ENVS="${NUM_ENVS}" \
  STEPS_PER_ENV="${STEPS_PER_ENV}" \
  ITERS=1 \
  MOTIONS=4 \
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
  SAVE_FREQUENCY=999999 \
  SAVE_LAST_FREQUENCY=999999 \
  NUM_CRITICS=2 \
  MULTI_CRITIC_ADVANTAGE_WEIGHTS='[0.5,0.5]' \
  DCPEFT_REWARD_CONTRACT="${CONTRACT}" \
  DCPEFT_REWARD_AUDIT_JSONL="${PROJECT_ROOT}/results/${RUN_NAME}_reward_contract.jsonl" \
  DCPEFT_CRITIC_DIAGNOSTICS=1 \
  DCPEFT_ACTOR_GRADIENT_DIAGNOSTICS=1 \
  DCPEFT_ACTOR_GRADIENT_NONMUTATING=1 \
  DCPEFT_ACTOR_GRADIENT_JSONL="${OUTPUT}" \
  DCPEFT_ACTOR_GRADIENT_CHECKPOINT="${CHECKPOINT}" \
  TRACKED_BODY_SET=source14 \
  ANTI_SHAKE_BODY_NAMES='[left_wrist_roll_link,right_wrist_roll_link]' \
  FOOT_CONTACT_LABEL_MODE=provided \
  ACTUATOR_RESPONSE_MODE=session0304_filter_only \
  ACTUATOR_RESPONSE_STRENGTH=1.0 \
  ACTUATOR_RESPONSE_IDEAL_FRACTION=0.5 \
  RESPONSE_AWARE_LORA_ENABLED=false \
  RESPONSE_AWARE_INPUT_SCALE=0.0 \
  PRESERVE_LORA_CHECKPOINT=true \
  STACK_LORA_CHECKPOINT_RESIDUAL=true \
  ACTOR_LORA_PREFIXES='[actor_module.decoders.g1_dyn]' \
  ACTOR_OUTPUT_MASK_ENABLED=true \
  ACTOR_OUTPUT_MASK_LAYERS='[actor_module.decoders.g1_dyn.module.12]' \
  ACTOR_OUTPUT_MASK_TOKENS='[_joint]' \
  ACTOR_OUTPUT_UNSELECTED_SCALE=0.0 \
  ACTOR_DECODER_DELTA_REPLAY=true \
  TRAIN_BOUNDARY_KEYS='[]' \
  TRAIN_ACTOR_LORA=true \
  TRAIN_CRITIC_LORA=true \
  SEPARATE_ACTOR_CRITIC_LR=true \
  ACTOR_LEARNING_RATE=1.0e-12 \
  CRITIC_LEARNING_RATE=1.0e-12 \
  ADAPTIVE_LR_MIN=1.0e-12 \
  ADAPTIVE_LR_MAX=1.0e-12 \
  PPO_EPOCHS=1 \
  NUM_MINI_BATCHES=4 \
  TRAINING_DIAGNOSTICS_ENABLED=true \
  ADAPTIVE_SAMPLING_ENABLED=false \
  TRAIN_ONLY_EVENTS='[]' \
  CLEAN_EVAL=true \
  CLEAN_COMMAND_INIT=true \
  USE_PAIRED_MOTIONS=true \
  START_FROM_FIRST_FRAME=false \
  FREEZE_FRAME_AUG=true \
  SOURCE_CANONICAL_REFERENCE=true \
  EE_WRIST_LINK=roll \
  REWARD_WRIST_LINK=roll \
  REWARD_POINT_MODE=source3 \
  X2_PD_PROFILE=foundation_stiff_lower \
  X2_ACTION_SCALE_PROFILE=foundation_hybrid_mobility2x \
  X2_COLLISION_PROFILE=mesh \
  X2_SELF_COLLISIONS=true \
  ANCHOR_POS_REWARD_WEIGHT=0.20 \
  ANCHOR_ORI_REWARD_WEIGHT=0.5 \
  RELATIVE_BODY_POS_REWARD_WEIGHT=1.0 \
  VR_TRACKING_REWARD_WEIGHT=2.0 \
  FOOT_CONTACT_PHASE_REWARD_WEIGHT=1.0 \
  FOOT_CONTACT_CONTINUOUS_LOAD=true \
  FOOT_CONTACT_CONTINUOUS_LOAD_MODE=weight_transfer \
  FOOT_CONTACT_PREPARATION_FUTURE_FRAME_INDEX=2 \
  FOOT_CONTACT_PREPARATION_PRESERVE_DOUBLE_CONTACT=false \
  SINGLE_SUPPORT_TRACKING_PREPARATION_FUTURE_FRAME_INDEX=0 \
  SINGLE_SUPPORT_COM_PREPARATION_FUTURE_FRAME_INDEX=0 \
  SWING_FOOT_HEIGHT_REWARD_WEIGHT=1.0 \
  SWING_FOOT_HEIGHT_REWARD_STD=0.025 \
  SOURCE_RETENTION_ENABLED=false \
  SOURCE_RETENTION_FIXED_REPLAY_ENABLED=false \
  bash "${BASE_LAUNCHER}"
