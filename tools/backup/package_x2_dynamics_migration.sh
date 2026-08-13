#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
LEGACY_ROOT=/home/humanplus/x2_teleop_final/x2_sonic
SANDBOX_ROOT=${LEGACY_ROOT}/sonic_x2_sandbox
OUT_DIR=${1:-${PROJECT_ROOT}/backups/migration_20260810}
OUT=${OUT_DIR}/x2_forward4_dynamics_feasibility_seed_v1.tar.gz
OVERLAY_DIR=${OUT_DIR}/migration_overlay

mkdir -p "${OUT_DIR}" "${OVERLAY_DIR}"

git -C "${SANDBOX_ROOT}" rev-parse HEAD > "${OVERLAY_DIR}/sonic_x2_sandbox_base_commit.txt"
git -C "${SANDBOX_ROOT}" diff --binary \
  --output="${OVERLAY_DIR}/sonic_x2_sandbox_tracked_changes.patch"
git -C "${SANDBOX_ROOT}" ls-files --others --exclude-standard -z \
  | tar -C "${SANDBOX_ROOT}" --null -czf \
      "${OVERLAY_DIR}/sonic_x2_sandbox_untracked_files.tar.gz" --files-from -

tar -C /home/humanplus -czf "${OUT}" \
  x2_teleop_final/assets/agibot_x2 \
  x2_teleop_final/x2_sonic/isaaclab_usd_cache \
  x2_teleop_final/x2_sonic/scripts \
  x2_teleop_final/x2_sonic/tools \
  x2_teleop_final/x2_sonic/tests \
  x2_teleop_final/x2_sonic/any2any_x2_bridge.py \
  x2_teleop_final/x2_sonic/any2any_x2_policy_adapter.py \
  x2_teleop_final/x2_sonic/any2any_x2_trainable_adapter.py \
  x2_teleop_final/x2_sonic/numpy_pickle_compat.py \
  x2_teleop_final/x2_sonic/official_x2_qpos.py \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/model_step_000200.pt \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/config.yaml \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/meta.yaml \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/training_diagnostics.jsonl \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/console.log \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/any2any_lora_trainable_report.json \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/lora_residual_preservation_report.json \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/lora_residual_stack_report.json \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/policy_partial_load_report.json \
  x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/value_model_partial_load_report.json \
  x2_teleop_final/x2_sonic/motion_lib_x2/stage72_official_true_forward4_v1 \
  x2_teleop_final/x2_sonic/data/processed/x2_real_readonly_session04_canonical_31dof.npz \
  projects/ZHY/CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_ideal_teleop_r1_s260 \
  projects/ZHY/CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_filter_teleop_r1_s260 \
  projects/ZHY/CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_delay_teleop_r1_s260 \
  projects/ZHY/CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_noise_teleop_r1_s260 \
  projects/ZHY/CWI_CrossEmbodiment_Sim/reports/baseline_report.md \
  projects/ZHY/CWI_CrossEmbodiment_Sim/reports/b0_r1_four_domain_gate.md \
  projects/ZHY/CWI_CrossEmbodiment_Sim/reports/b0_r1_four_domain_gate.json \
  projects/ZHY/CWI_CrossEmbodiment_Sim/docs/backup/X2_DYNAMICS_FEASIBILIZATION_MIGRATION_CARD.md \
  -C "${OUT_DIR}" migration_overlay

sha256sum "${OUT}" > "${OUT}.sha256"
tar -tzf "${OUT}" > "${OUT}.contents.txt"

echo "archive=${OUT}"
stat --format='bytes=%s' "${OUT}"
cat "${OUT}.sha256"
