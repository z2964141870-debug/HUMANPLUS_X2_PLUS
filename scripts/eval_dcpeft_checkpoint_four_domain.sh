#!/usr/bin/env bash
set -euo pipefail

# Evaluate a saved B1/E1/N1 checkpoint with the frozen B0 four-domain panel.
# Usage: bash scripts/eval_dcpeft_checkpoint_four_domain.sh {B1|E1|N1} ITERS [SEED]

PROJECT_ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
SOURCE_ROOT=/home/humanplus/x2_teleop_final/x2_sonic
ARM=$(printf '%s' "${1:?arm required: B1|E1|N1}" | tr '[:lower:]' '[:upper:]')
ITERS="${2:?training iterations required}"
SEED="${3:-0}"
STEPS="${STEPS:-260}"
RUN_ID="${RUN_ID:-v1}"
MOTION="${SOURCE_ROOT}/motion_lib_x2/stage72_official_true_forward4_v1"
LOWER_ARM=$(printf '%s' "${ARM}" | tr '[:upper:]' '[:lower:]')
CHECKPOINT="${PROJECT_ROOT}/logs/dcpeft_${LOWER_ARM}_from_stage152b_i${ITERS}_seed${SEED}_${RUN_ID}/model_step_$(printf '%06d' "${ITERS}").pt"

case "${ARM}" in
  B1)
    NUM_CRITICS=1
    ADVANTAGE_WEIGHTS=null
    REWARD_CONTRACT=
    ;;
  E1)
    NUM_CRITICS=2
    ADVANTAGE_WEIGHTS='[0.5,0.5]'
    REWARD_CONTRACT="${PROJECT_ROOT}/configs/reward_contract_dual_v1.json"
    ;;
  N1)
    NUM_CRITICS=2
    ADVANTAGE_WEIGHTS='[0.5,0.5]'
    REWARD_CONTRACT="${PROJECT_ROOT}/configs/reward_contract_bad_materialized_v1.json"
    ;;
  *)
    echo "usage: $0 {B1|E1|N1} ITERS [SEED]" >&2
    exit 2
    ;;
esac

[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }

PANEL_NAME="dcpeft_${LOWER_ARM}_i${ITERS}"
LABEL="${LOWER_ARM}_i${ITERS}"
TAG="seed${SEED}_${RUN_ID}"
DOMAINS=(ideal filter delay noise)

trace_path() {
  local domain=$1
  echo "${PROJECT_ROOT}/logs/x2_stage19_${PANEL_NAME}_${LABEL}_${domain}_teleop_${TAG}_s${STEPS}/x2_wrist_rollout_trace.jsonl"
}

for domain in "${DOMAINS[@]}"; do
  if [[ -s "$(trace_path "${domain}")" ]]; then
    continue
  fi
  env \
    NUM_CRITICS="${NUM_CRITICS}" \
    MULTI_CRITIC_ADVANTAGE_WEIGHTS="${ADVANTAGE_WEIGHTS}" \
    DCPEFT_REWARD_CONTRACT="${REWARD_CONTRACT}" \
    DCPEFT_REWARD_AUDIT_JSONL="${PROJECT_ROOT}/results/${PANEL_NAME}_${domain}_${TAG}_reward_contract.jsonl" \
    CHECKPOINT_OVERRIDE="${CHECKPOINT}" \
    MOTION_FILE_VALUE="${MOTION}" \
    NUM_ENVS_VALUE=4 MOTIONS_VALUE=4 SEED_VALUE="${SEED}" \
    PANEL_NAME="${PANEL_NAME}" PANEL_TAG="${TAG}" \
    EXP_CONFIG_VALUE=manager/universal_token/all_modes/sonic_x2_phuma_response_context \
    RESPONSE_AWARE_LORA_ENABLED_VALUE=false \
    RESPONSE_AWARE_INPUT_SCALE_VALUE=0.0 \
    ACTOR_OUTPUT_MASK_ENABLED_VALUE=true \
    ACTOR_OUTPUT_MASK_LAYERS_VALUE='[actor_module.decoders.g1_dyn.module.12]' \
    ACTOR_OUTPUT_MASK_TOKENS_VALUE='[_joint]' \
    X2_PD_PROFILE_VALUE=foundation_stiff_lower \
    X2_ACTION_SCALE_PROFILE_VALUE=foundation_hybrid_mobility2x \
    X2_COLLISION_PROFILE_VALUE=mesh \
    SIM_DT_VALUE=0.005 DECIMATION_VALUE=4 \
    FOOT_CONTACT_LABEL_MODE=provided \
    PRESERVE_LORA_CHECKPOINT_VALUE=true \
    STACK_LORA_CHECKPOINT_RESIDUAL=true \
    bash "${PROJECT_ROOT}/scripts/run_b0_stage19_panel.sh" "${LABEL}" "${domain}" "${STEPS}"
done

args=()
for domain in "${DOMAINS[@]}"; do
  args+=(--trace "${domain}=$(trace_path "${domain}")")
done

python "${SOURCE_ROOT}/tools/analyze_x2_locomotion_progress.py" \
  "${args[@]}" \
  --min-reference-displacement-m 0.10 \
  --report-json "${PROJECT_ROOT}/reports/${LOWER_ARM}_i${ITERS}_seed${SEED}_four_domain_gate.json" \
  --report-md "${PROJECT_ROOT}/reports/${LOWER_ARM}_i${ITERS}_seed${SEED}_four_domain_gate.md"

echo "dcpeft_four_domain=OK arm=${ARM} iters=${ITERS} seed=${SEED}"
