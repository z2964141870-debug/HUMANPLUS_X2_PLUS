#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
SOURCE="$ROOT/artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt"
REFERENCE="$ROOT/artifacts/retarget/x2_phase69_reward_attribution/rollout_evidence_rerun1.pt"
BASE_REPORT="$ROOT/reports/retarget/x2_phase70_long_lookahead"
REPORT="${BASE_REPORT}_rerun1"
PREREG="${BASE_REPORT}_prereg_v4.json"
EVIDENCE="$ROOT/artifacts/retarget/x2_phase70_long_lookahead/rollout_evidence_rerun1.pt"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_phase70_long_lookahead.py"
PYTHON=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
LOG=/tmp/x2_phase70_logs/long_lookahead_rerun1.log

outputs=(
  "${REPORT}_screen.json"
  "${REPORT}_resource.json"
  "${REPORT}_result.json"
  "${REPORT}_result.json.sha256"
  "${REPORT}.md"
  "$EVIDENCE"
  "${EVIDENCE}.tmp"
  "$LOG"
)
for path in "${outputs[@]}"; do
  [[ ! -e "$path" ]] || {
    echo "refusing to overwrite immutable Phase70 output: $path" >&2
    exit 2
  }
done
free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 20 * 1024 * 1024 )) || {
  echo "Phase70 requires at least 20 GiB free" >&2
  exit 2
}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25
export CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
mkdir -p "$(dirname "$EVIDENCE")" "$(dirname "$LOG")"

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" -m pytest -q \
  "$ROOT/tests/test_phase70_long_lookahead.py" \
  "$ROOT/tests/test_phase70_long_lookahead_finalizer.py" \
  "$ROOT/tests/test_phase69_reward_attribution.py" \
  "$ROOT/tests/test_phase68_residual_ppo.py" \
  "$ROOT/tests/test_phase67_phase_conditioned_residual.py"

set +e
CWI_PHASE70_LONG_LOOKAHEAD=1 python3 "$LEDGER" \
  --label phase70_repair1_fresh_rng_zero_optimizer_long_lookahead \
  --resource-output "${REPORT}_resource.json" \
  --log "$LOG" --disk-path "$ROOT" -- \
  "$PYTHON" "$RUNNER" \
  --mode train --checkpoint "$ORIGINAL" \
  --residual-checkpoint "$SOURCE" \
  --reference-rollout-bundle "$REFERENCE" \
  --phase70-prereg "$PREREG" \
  --rollout-bundle "$EVIDENCE" --output "${REPORT}_screen.json" \
  --num-envs 64 --seed 42 --eval-steps 400 --device cuda:0 --headless
runner_status=$?
set -e
if [[ "$runner_status" -ne 0 || ! -f "${REPORT}_screen.json" ]]; then
  echo "Phase70 runner did not produce a screen (status=$runner_status)" >&2
  exit 1
fi
if ! jq -e '
  .schema == "x2_phase70_long_lookahead_screen_v1"
  and .decision == "LOOKAHEAD_VALID_PENDING_FINALIZATION"
' "${REPORT}_screen.json" >/dev/null; then
  echo "Phase70 screen is absent or not valid-pending" >&2
  exit 1
fi
if ! jq -e '.exit_code == 0' "${REPORT}_resource.json" >/dev/null; then
  echo "Phase70 resource ledger did not record exit_code=0" >&2
  exit 1
fi

python3 "$FINALIZER" \
  --prereg "$PREREG" \
  --screen "${REPORT}_screen.json" \
  --resource "${REPORT}_resource.json" \
  --output "${REPORT}_result.json" \
  --markdown "${REPORT}.md"
