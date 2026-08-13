#!/usr/bin/env bash
set -euo pipefail

EXECUTE=false
[[ "${1:-}" == "--execute" ]] && EXECUTE=true
ROOT="${X2_MIGRATION_ROOT:-${HOME}/x2_migration_20260812}"
ZHY_ROOT="${X2_ZHY_ROOT:-${HOME}/projects/ZHY}"
HUMAN_ROOT="${X2_HUMAN_PROJECT_ROOT:-${HOME}/projects/Human+智能服装动作捕捉系统}"
REPO="${ZHY_ROOT}/CWI_CrossEmbodiment_Sim"
D="${ROOT}/downloads"

cat <<EOF
Restore plan:
  migration downloads: ${D}
  ZHY workspace:       ${ZHY_ROOT}
  main repository:     ${REPO}
  GMR runtime:         ${HUMAN_ROOT}/general_motion_retargeting
  x2 teleop seed:      ${HOME}/x2_teleop_final
EOF

if [[ "${EXECUTE}" != true ]]; then
  echo "Dry-run only. Re-run with --execute after reviewing paths."
  exit 0
fi

"$(dirname "$0")/verify_x2_migration.sh" --core
mkdir -p "${ROOT}/restore" "${ZHY_ROOT}" "${HUMAN_ROOT}" "${HOME}/humanoid-GPT"

cycle10_dir="${ROOT}/restore/cycle10"
mkdir -p "${cycle10_dir}"
tar -xzf "${D}/cycles/x2_cycle10_dynamic_base_20260811.tar.gz" -C "${cycle10_dir}"
bundle="${cycle10_dir}/x2_cycle10_20260811/HUMANPLUS_X2_PLUS_cycle10.bundle"
if [[ ! -d "${REPO}/.git" ]]; then
  git clone "${bundle}" "${REPO}"
  git -C "${REPO}" switch main
else
  head="$(git -C "${REPO}" rev-parse HEAD)"
  [[ "${head}" == "ac4d34f97a461b3e5c61bbec754ddef5f89449da" ]] || {
    echo "existing repo HEAD is not the frozen baseline; refusing overlay: ${head}" >&2
    exit 8
  }
fi

tar -xzf "${D}/cycles/x2_current_worktree_delta_phase40_20260812.tar.gz" -C "${REPO}"
mkdir -p "${REPO}/docs/backup"
cp -a "$(dirname "$0")" "${REPO}/docs/backup/new_pc_bdpan"
tar -xzf "${D}/seed/x2_forward4_dynamics_feasibility_seed_v1.tar.gz" -C "${HOME}"
tar -xzf "${D}/source_trees/x2_official_rl_deploy_v1_source_assets_20260810.tar.gz" -C "${ZHY_ROOT}"
tar -xzf "${D}/source_trees/general_motion_retargeting_actual_runtime_20260810.tar.gz" -C "${HUMAN_ROOT}"
tar -xzf "${D}/source_trees/Humanoid-GPT_requested_tree_20260810.tar.gz" -C "${HOME}/humanoid-GPT"
tar -xzf "${D}/dynamic_retargeting/x2_dynamic_retargeting_race_full_20260811.tar.gz" -C "${ZHY_ROOT}"

echo "Core restore complete. Phase archives remain downloaded but are not bulk-extracted"
echo "because their historical roots differ; inspect each archive before selective restore."
echo "Next: rebuild environment from ${REPO}/docs/backup/X2_MIGRATION_FINAL_REQUIREMENTS_20260810.md"
