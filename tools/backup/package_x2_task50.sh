#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="$OFFICIAL_ROOT/results/official_native_strict_20260807"
OUTPUT="${OUTPUT:-$REPO_ROOT/backups/x2_official_upper_pd_task50_20260808.tar.gz}"
STAGING="$(mktemp -d /tmp/x2-task50.XXXXXX)"
trap 'rm -rf "$STAGING"' EXIT

mkdir -p \
  "$STAGING/models" "$STAGING/assets/official_x2" \
  "$STAGING/official_results" "$STAGING/reports" \
  "$STAGING/tools/official_x2" "$STAGING/tools"

cp "$OFFICIAL_ROOT/models/stage219_s2600_actor.onnx" "$STAGING/models/"
cp "$REPO_ROOT/assets/official_x2/upper_swing_arms_stand_14dof.npz" "$STAGING/assets/official_x2/"
cp "$REPO_ROOT/assets/official_x2/upper_wave_real_14dof.npz" "$STAGING/assets/official_x2/"

for pattern in \
  'stage251_upper_*_straight_r[123].json' \
  'stage252_upper_*_right_r[123].json' \
  'stage252_upper_*_left_r[123].json' \
  'stage253_left_yawg2p5_*_r[123].json' \
  'stage254_*.json' \
  'stage255_*.json' \
  'stage256_*.json' \
  'stage257_*_straight_r[123].json' \
  'stage258_*.json' \
  'stage259_*.json' \
  'stage260_*.json' \
  'stage261_*.json' \
  'stage262_*.json' \
  'stage263_*_straight_r[123].json'; do
  matches=("$RESULT_ROOT"/$pattern)
  [[ -e "${matches[0]}" ]] || { echo "missing official result pattern: $pattern" >&2; exit 2; }
  cp "${matches[@]}" "$STAGING/official_results/"
done

for stage in 251 252 253 257 263; do
  matches=("$REPO_ROOT"/reports/official_x2/stage${stage}_*)
  [[ -e "${matches[0]}" ]] || { echo "missing Stage${stage} report" >&2; exit 2; }
  cp "${matches[@]}" "$STAGING/reports/"
done
cp "$REPO_ROOT/reports/official_x2/stage264_longtrain_readiness_task50.json" "$STAGING/reports/"
cp "$REPO_ROOT/reports/official_x2/stage264_longtrain_readiness_task50.md" "$STAGING/reports/"

for name in \
  stage208_official_mujoco_adapter.py \
  run_official_gate_case.sh \
  run_official_gate_inner.sh \
  run_stage251_upper_straight_panel.sh \
  run_stage252_upper_turn_panel.sh \
  run_stage253_left_yaw_gain_panel.sh \
  run_stage257_pd_upper_straight_panel.sh \
  summarize_stage251_upper_disturbance.py \
  summarize_stage252_upper_turn.py \
  summarize_stage253_left_yaw_gain.py \
  summarize_stage257_pd_upper_straight.py \
  summarize_stage264_longtrain_readiness.py; do
  cp "$REPO_ROOT/tools/official_x2/$name" "$STAGING/tools/official_x2/"
done
cp "$REPO_ROOT/tools/export_upper_motion_npz.py" "$STAGING/tools/"

(
  cd "$STAGING"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
tar -C "$STAGING" -czf "$OUTPUT" .
sha256sum "$OUTPUT"
