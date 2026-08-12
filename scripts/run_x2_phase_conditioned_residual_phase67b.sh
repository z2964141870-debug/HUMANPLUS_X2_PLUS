#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
REPORTDIR="$ROOT/reports/retarget"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_phase_conditioned_residual_phase67b.py"
PREREG="$REPORTDIR/x2_phase_conditioned_residual_phase67b_prereg.json"
SCREEN="$REPORTDIR/x2_phase_conditioned_residual_phase67b_screen.json"
RESOURCE="$REPORTDIR/x2_phase_conditioned_residual_phase67b_resource.json"
RESULT="$REPORTDIR/x2_phase_conditioned_residual_phase67b_result.json"
MARKDOWN="$REPORTDIR/x2_phase_conditioned_residual_phase67b.md"
LOGDIR=/tmp/x2_phase_conditioned_residual_phase67b_logs

for path in "$SCREEN" "$RESOURCE" "$RESULT" "$MARKDOWN"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite immutable Phase67b output: $path" >&2; exit 2; }
done
free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 20 * 1024 * 1024 )) || { echo "Phase67b requires at least 20 GiB free" >&2; exit 2; }

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=1.0
export CWI_UPPER_DETERMINISTIC_SPLIT=0 CWI_UPPER_SPLIT_MODE=contiguous
export CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
export CWI_PHASE67_RESIDUAL_LIVE_ZERO=1
mkdir -p "$REPORTDIR" "$LOGDIR"

PYTHONPATH="$ROOT/src" conda run -n x2-sonic-isaaclab python -m pytest -q \
  "$ROOT/tests/test_phase67_phase_conditioned_residual.py"

if ! python3 "$LEDGER" \
  --label phase67b_fail_closed_suffix_live_zero \
  --resource-output "$RESOURCE" --log "$LOGDIR/live_zero.log" --disk-path "$ROOT" -- \
  conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
  --mode screen --checkpoint "$ORIGINAL" --output "$SCREEN" \
  --num-envs 64 --seed 42 --eval-steps 200 --device cuda:0 --headless; then
  tail -n 160 "$LOGDIR/live_zero.log" >&2
  exit 1
fi

python3 "$FINALIZER" --input "$SCREEN" --resource "$RESOURCE" --prereg "$PREREG" \
  --output "$RESULT" --markdown "$MARKDOWN"
