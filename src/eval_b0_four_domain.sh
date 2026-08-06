#!/usr/bin/env bash
# DC-PEFT Phase 0: B0 基线复跑（零训练，deterministic eval）
# 复刻 scripts/eval_x2_stage152_response_aware_lora_ablation.sh 的 B 组参数，
# 区别仅在：PANEL_NAME/PANEL_TAG 独立命名（不覆盖历史 trace），
# 分析报告写入新工程 results/（不触碰旧工程 docs/reports）。
set -euo pipefail

ROOT=/home/humanplus/x2_teleop_final/x2_sonic
NEW=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
RUN_TAG="${1:?usage: $0 <r1|r2> [domains...]}"; shift || true
DOMAINS=("${@:-ideal filter delay noise}")
if [[ $# -eq 0 ]]; then DOMAINS=(ideal filter delay noise); fi

CHECKPOINT=${ROOT}/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/model_step_000200.pt
MOTION=${ROOT}/motion_lib_x2/stage72_official_true_forward4_v1
STEPS=260
PANEL_NAME=dcpeft_p0
PANEL_TAG=B_pilot_seed0_${RUN_TAG}
LABEL=stage152_B_pilot

[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint" >&2; exit 2; }

trace_path() {
  echo "${ROOT}/logs/ppo_dryrun/x2_stage19_${PANEL_NAME}_${LABEL}_$1_teleop_${PANEL_TAG}_s${STEPS}/x2_wrist_rollout_trace.jsonl"
}

for domain in "${DOMAINS[@]}"; do
  if [[ -s "$(trace_path "${domain}")" ]]; then
    echo "skip existing trace: ${domain}"
    continue
  fi
  env \
    CHECKPOINT_OVERRIDE="${CHECKPOINT}" \
    MOTION_FILE_VALUE="${MOTION}" \
    NUM_ENVS_VALUE=4 MOTIONS_VALUE=4 SEED_VALUE=0 \
    PANEL_NAME="${PANEL_NAME}" PANEL_TAG="${PANEL_TAG}" \
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
    bash "${ROOT}/scripts/run_x2_stage19_foot_panel.sh" "${LABEL}" "${domain}" "${STEPS}"
done

args=()
for domain in "${DOMAINS[@]}"; do
  args+=(--trace "${domain}=$(trace_path "${domain}")")
done

mkdir -p "${NEW}/results/phase0_b0"
python "${ROOT}/tools/analyze_x2_locomotion_progress.py" \
  "${args[@]}" --min-reference-displacement-m 0.10 \
  --report-json "${NEW}/results/phase0_b0/b0_four_domain_gate_${RUN_TAG}.json" \
  --report-md "${NEW}/results/phase0_b0/b0_four_domain_gate_${RUN_TAG}.md"

echo "dcpeft_p0_b0_eval=OK tag=${RUN_TAG}"
