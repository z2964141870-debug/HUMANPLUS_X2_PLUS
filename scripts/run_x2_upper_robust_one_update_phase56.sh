#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/humanplus/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
OUTDIR="$ROOT/artifacts/retarget/x2_upper_robust_lower_phase56"
REPORTDIR="$ROOT/reports/retarget"
COMMON=(--num-envs 64 --seed 42 --device cuda:0 --headless)
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=0.50
export CWI_UPPER_DETERMINISTIC_SPLIT=1 CWI_UPPER_SPLIT_MODE=interleaved
export CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
mkdir -p "$OUTDIR" "$REPORTDIR"

conda run --no-capture-output -n x2-sonic-isaaclab python \
  "$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py" \
  --mode eval --checkpoint "$ORIGINAL" --eval-steps 200 \
  --output "$REPORTDIR/x2_upper_robust_lower_phase56_pre.json" "${COMMON[@]}"
test -s "$REPORTDIR/x2_upper_robust_lower_phase56_pre.json"

conda run --no-capture-output -n x2-sonic-isaaclab python \
  "$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py" \
  --mode train --checkpoint "$ORIGINAL" \
  --source-output "$OUTDIR/source_stage219.pt" \
  --final-output "$OUTDIR/final_one_update.pt" \
  --output "$REPORTDIR/x2_upper_robust_lower_phase56_train.json" "${COMMON[@]}"
test -s "$REPORTDIR/x2_upper_robust_lower_phase56_train.json"
test -s "$OUTDIR/source_stage219.pt"
test -s "$OUTDIR/final_one_update.pt"

conda run --no-capture-output -n x2-sonic-isaaclab python \
  "$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py" \
  --mode eval --checkpoint "$OUTDIR/final_one_update.pt" --eval-steps 200 \
  --output "$REPORTDIR/x2_upper_robust_lower_phase56_post.json" "${COMMON[@]}"
test -s "$REPORTDIR/x2_upper_robust_lower_phase56_post.json"

PYTHONPATH="$ROOT/src" conda run -n x2-sonic-isaaclab python \
  "$ROOT/tools/retarget/finalize_x2_upper_robust_phase56.py" \
  --pre "$REPORTDIR/x2_upper_robust_lower_phase56_pre.json" \
  --train "$REPORTDIR/x2_upper_robust_lower_phase56_train.json" \
  --post "$REPORTDIR/x2_upper_robust_lower_phase56_post.json" \
  --prereg "$REPORTDIR/x2_upper_robust_lower_one_update_phase56_prereg.json" \
  --output "$REPORTDIR/x2_upper_robust_lower_one_update_phase56.json"
