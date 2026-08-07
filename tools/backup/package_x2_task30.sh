#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="$OFFICIAL_ROOT/results/official_native_strict_20260807"
OUTPUT="${OUTPUT:-$REPO_ROOT/backups/x2_official_state_transport_task30_20260808.tar.gz}"
STAGING="$(mktemp -d /tmp/x2-task30.XXXXXX)"
trap 'rm -rf "$STAGING"' EXIT

mkdir -p "$STAGING/models" "$STAGING/official_results" "$STAGING/direct_results" "$STAGING/reports"
cp "$OFFICIAL_ROOT/models/stage219_s2600_actor.onnx" "$STAGING/models/"
cp "$OFFICIAL_ROOT/models/stand_backend_scratch_i150_actor.onnx" "$STAGING/models/"

for pattern in \
  'stage232_stateqos1_heading050_straight_r*.json' \
  'stage231_stateqos1_gate_turn_right_r*.json' \
  'stage231_stateqos1_gate_turn_left_r*.json' \
  'stage233_stateqos1_heading_recovery_straight_r*.json' \
  'stage230_stateqos10_timing_straight_r1.json' \
  'stage226_statepredict2ms_straight_r1.json' \
  'stage226_statepredict4ms_straight_r1.json'; do
  matches=("$RESULT_ROOT"/$pattern)
  [[ -e "${matches[0]}" ]] || { echo "missing official result pattern: $pattern" >&2; exit 2; }
  cp "${matches[@]}" "$STAGING/official_results/"
done

for name in \
  stage223_direct_mujoco_closed_loop_straight_r2.json \
  stage223_stage220_s2700_direct_mujoco_straight_r1.json \
  stage224_stage219_direct_obslag0p2.json \
  stage225b_stage219_direct_obslag0p2_predict_acc.json \
  stage225b_stage219_direct_obslag0p25_predict_acc.json; do
  cp "$REPO_ROOT/reports/official_x2/$name" "$STAGING/direct_results/"
done

cp "$REPO_ROOT/reports/official_x2/stage225_vendor_mjcf_delay_prediction_20260808.md" "$STAGING/reports/"
cp "$REPO_ROOT/reports/official_x2/stage232_state_transport_sweet_point_20260808.json" "$STAGING/reports/"
cp "$REPO_ROOT/reports/official_x2/stage232_state_transport_sweet_point_20260808.md" "$STAGING/reports/"

(
  cd "$STAGING"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
tar -C "$STAGING" -czf "$OUTPUT" .
sha256sum "$OUTPUT"
