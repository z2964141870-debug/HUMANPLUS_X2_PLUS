#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/home/humanplus/x2_teleop_final/x2_sonic
PROJECT_ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
CHECKPOINT=${CHECKPOINT:-${SOURCE_ROOT}/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/model_step_000200.pt}
MOTION=${MOTION:-${SOURCE_ROOT}/motion_lib_x2/stage72_official_true_forward4_v1}
REPEAT=${REPEAT:-r1}
STEPS=${STEPS:-260}
SEED=${SEED:-0}
DOMAINS=(ideal filter delay noise)

trace_path() {
  local domain=$1
  echo "${PROJECT_ROOT}/logs/x2_stage19_dcpeft_b0_stage152_b0_${domain}_teleop_${REPEAT}_s${STEPS}/x2_wrist_rollout_trace.jsonl"
}

for domain in "${DOMAINS[@]}"; do
  if [[ -s "$(trace_path "${domain}")" ]]; then
    continue
  fi
  env \
    CHECKPOINT_OVERRIDE="${CHECKPOINT}" \
    MOTION_FILE_VALUE="${MOTION}" \
    NUM_ENVS_VALUE=4 MOTIONS_VALUE=4 SEED_VALUE="${SEED}" \
    PANEL_NAME=dcpeft_b0 PANEL_TAG="${REPEAT}" \
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
    bash "${PROJECT_ROOT}/scripts/run_b0_stage19_panel.sh" stage152_b0 "${domain}" "${STEPS}"
done

args=()
for domain in "${DOMAINS[@]}"; do
  args+=(--trace "${domain}=$(trace_path "${domain}")")
done

python "${SOURCE_ROOT}/tools/analyze_x2_locomotion_progress.py" \
  "${args[@]}" \
  --min-reference-displacement-m 0.10 \
  --report-json "${PROJECT_ROOT}/reports/b0_${REPEAT}_four_domain_gate.json" \
  --report-md "${PROJECT_ROOT}/reports/b0_${REPEAT}_four_domain_gate.md"

echo "dcpeft_b0_four_domain=OK repeat=${REPEAT}"
