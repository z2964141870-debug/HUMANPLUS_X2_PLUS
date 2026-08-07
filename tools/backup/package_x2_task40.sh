#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="$OFFICIAL_ROOT/results/official_native_strict_20260807"
OUTPUT="${OUTPUT:-$REPO_ROOT/backups/x2_official_rsl_contract_task40_20260808.tar.gz}"
STAGING="$(mktemp -d /tmp/x2-task40.XXXXXX)"
trap 'rm -rf "$STAGING"' EXIT

mkdir -p \
  "$STAGING/models" "$STAGING/official_results" "$STAGING/reports" \
  "$STAGING/tools/official_x2" "$STAGING/videos"
cp "$OFFICIAL_ROOT/models/stage219_s2600_actor.onnx" "$STAGING/models/"

for pattern in \
  'stage244_rsl_actor_clip_medium_straight_r*.json' \
  'stage246_rsl_actor_clip_turn_progress_right_r*.json' \
  'stage247_rsl_actor_clip_turn_progress_left_mirrorfix_r*.json' \
  'stage248_rsl_actor_clip_vx025_native_straight_r*.json' \
  'stage249_rsl_actor_clip_vx025_turn_progress_right_r*.json' \
  'stage249_rsl_actor_clip_vx025_turn_progress_left_r*.json' \
  'stage243_vx025_quality_waist085_straight_r*.json' \
  'stage245_rsl_actor_clip_medium_turn_right_r*.json' \
  'stage245_rsl_actor_clip_medium_turn_left_r*.json'; do
  matches=("$RESULT_ROOT"/$pattern)
  [[ -e "${matches[0]}" ]] || { echo "missing official result pattern: $pattern" >&2; exit 2; }
  cp "${matches[@]}" "$STAGING/official_results/"
done

for name in \
  stage242_low_speed_control_contract_sweet_point_20260808.json \
  stage242_low_speed_control_contract_sweet_point_20260808.md \
  stage250_rsl_action_contract_fix_20260808.json \
  stage250_rsl_action_contract_fix_20260808.md; do
  cp "$REPO_ROOT/reports/official_x2/$name" "$STAGING/reports/"
done

for name in \
  stage208_official_mujoco_adapter.py \
  analyze_actor_raw_action.py \
  analyze_official_motion_quality.py \
  run_official_gate_case.sh \
  run_official_gate_inner.sh \
  summarize_stage250_action_contract_fix.py \
  record_stage250_official_case.sh \
  make_stage250_before_after_video.sh; do
  cp "$REPO_ROOT/tools/official_x2/$name" "$STAGING/tools/official_x2/"
done

cp "$REPO_ROOT/videos/official_x2/stage250/stage219_v5_vs_stage250_official_mujoco_three_skill.mp4" "$STAGING/videos/"

(
  cd "$STAGING"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
tar -C "$STAGING" -czf "$OUTPUT" .
sha256sum "$OUTPUT"
