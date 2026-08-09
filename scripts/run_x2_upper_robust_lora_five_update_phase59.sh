#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/humanplus/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
OUTDIR="$ROOT/artifacts/retarget/x2_upper_robust_lower_lora_phase59"
REPORTDIR="$ROOT/reports/retarget"
STATE="$REPORTDIR/x2_upper_robust_lower_lora_phase59_state.json"
COMMON=(--num-envs 64 --seed 42 --device cuda:0 --headless)
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_PHASE59_PEFT=1
export CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=0.50
export CWI_UPPER_DETERMINISTIC_SPLIT=1 CWI_UPPER_SPLIT_MODE=interleaved
export CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
mkdir -p "$OUTDIR" "$REPORTDIR"
rm -f "$STATE"

PRE="$REPORTDIR/x2_upper_robust_lower_lora_phase59_pre.json"
conda run --no-capture-output -n x2-sonic-isaaclab python \
  "$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py" \
  --mode eval --checkpoint "$ORIGINAL" --eval-steps 200 --output "$PRE" "${COMMON[@]}"
test -s "$PRE"

CURRENT="$ORIGINAL"
for UPDATE in 1 2 3 4 5; do
  TAG=$(printf "%02d" "$UPDATE")
  if [[ "$UPDATE" -eq 1 ]]; then
    SOURCE_OUT="$OUTDIR/source_stage219_zero_lora.pt"
  else
    SOURCE_OUT="$OUTDIR/pre_working.pt"
  fi
  CANDIDATE="$OUTDIR/candidate_update${TAG}.pt"
  TRAIN_REPORT="$REPORTDIR/x2_upper_robust_lower_lora_phase59_update${TAG}_train.json"
  EVAL_REPORT="$REPORTDIR/x2_upper_robust_lower_lora_phase59_update${TAG}_eval.json"
  GATE_REPORT="$REPORTDIR/x2_upper_robust_lower_lora_phase59_update${TAG}_gate.json"
  conda run --no-capture-output -n x2-sonic-isaaclab python \
    "$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py" \
    --mode train --checkpoint "$CURRENT" --update-index "$UPDATE" \
    --source-output "$SOURCE_OUT" --final-output "$CANDIDATE" \
    --output "$TRAIN_REPORT" "${COMMON[@]}"
  test -s "$TRAIN_REPORT"
  test -s "$CANDIDATE"
  conda run --no-capture-output -n x2-sonic-isaaclab python \
    "$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py" \
    --mode eval --checkpoint "$CANDIDATE" --eval-steps 200 \
    --output "$EVAL_REPORT" "${COMMON[@]}"
  test -s "$EVAL_REPORT"
  PYTHONPATH="$ROOT/src" conda run -n x2-sonic-isaaclab python \
    "$ROOT/tools/retarget/finalize_x2_upper_robust_phase56.py" \
    --pre "$PRE" --train "$TRAIN_REPORT" --post "$EVAL_REPORT" \
    --prereg "$REPORTDIR/x2_upper_robust_lower_lora_five_update_phase59_prereg.json" \
    --output "$GATE_REPORT"
  set +e
  PYTHONPATH="$ROOT/src" conda run -n x2-sonic-isaaclab python \
    "$ROOT/tools/retarget/select_x2_upper_robust_phase59.py" \
    --gate "$GATE_REPORT" --candidate "$CANDIDATE" --state "$STATE" --update "$UPDATE"
  STATUS=$?
  set -e
  if [[ "$STATUS" -eq 10 ]]; then
    break
  fi
  if [[ "$STATUS" -ne 0 ]]; then
    exit "$STATUS"
  fi
  CURRENT="$CANDIDATE"
done

PYTHONPATH="$ROOT/src" conda run -n x2-sonic-isaaclab python \
  "$ROOT/tools/retarget/finalize_x2_upper_robust_phase59.py" \
  --state "$STATE" --artifact-dir "$OUTDIR" \
  --source "$OUTDIR/source_stage219_zero_lora.pt" \
  --output "$REPORTDIR/x2_upper_robust_lower_lora_five_update_phase59.json"
