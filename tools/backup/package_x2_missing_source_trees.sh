#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
OUT_DIR=${1:-${PROJECT_ROOT}/backups/migration_20260810/source_trees}
mkdir -p "${OUT_DIR}"

SDK_OUT=${OUT_DIR}/x2_official_rl_deploy_v1_source_assets_20260810.tar.gz
HGPT_OUT=${OUT_DIR}/Humanoid-GPT_requested_tree_20260810.tar.gz
GMR_OUT=${OUT_DIR}/general_motion_retargeting_actual_runtime_20260810.tar.gz

# Results/cache are already archived in the phase bundles and contain several
# gigabytes of mmap traces. Keep SDK source, vendor workspace, scene, YAML,
# meshes, models, Dockerfile, templates, variants and compiled workspace.
tar -C /home/humanplus/projects/ZHY -czf "${SDK_OUT}" \
  --exclude='x2_official_rl_deploy_v1/results' \
  --exclude='x2_official_rl_deploy_v1/cache' \
  --exclude='x2_official_rl_deploy_v1/downloads' \
  x2_official_rl_deploy_v1

# Preserve the exact directory requested by the migration operator, even
# though the audit shows it contains reports rather than the GMR package.
tar -C /home/humanplus/humanoid-GPT -czf "${HGPT_OUT}" Humanoid-GPT

# This is the package actually imported as general_motion_retargeting in the
# audited h-gpt environment. Include the adjacent gmr config shim as well.
tar -C '/home/humanplus/projects/Human+智能服装动作捕捉系统' -czf "${GMR_OUT}" \
  general_motion_retargeting gmr

for archive in "${SDK_OUT}" "${HGPT_OUT}" "${GMR_OUT}"; do
  sha256sum "${archive}" > "${archive}.sha256"
  tar -tzf "${archive}" > "${archive}.contents.txt"
  stat --format='%n %s bytes' "${archive}"
  cat "${archive}.sha256"
done
