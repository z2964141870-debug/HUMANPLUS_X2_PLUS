#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
REPORT="$ROOT/reports/retarget/x2_phase_conditioned_residual_phase68"
ARTIFACT="$ROOT/artifacts/retarget/x2_phase_conditioned_residual_phase68"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_phase_conditioned_residual_phase68.py"
LOGDIR=/tmp/x2_phase68_logs

outputs=(
  "${REPORT}_train.json" "${REPORT}_train_resource.json"
  "${REPORT}_source_eval.json" "${REPORT}_source_resource.json"
  "${REPORT}_candidate_eval.json" "${REPORT}_candidate_resource.json"
  "${REPORT}_result.json" "${REPORT}.md"
  "$ARTIFACT/source_stage219_zero_residual.pt"
  "$ARTIFACT/candidate_phase_conditioned_residual_one_step.pt"
)
for path in "${outputs[@]}"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite immutable Phase68 output: $path" >&2; exit 2; }
done
free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 20 * 1024 * 1024 )) || { echo "Phase68 requires at least 20 GiB free" >&2; exit 2; }

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25
export CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
mkdir -p "$ARTIFACT" "$LOGDIR"

PYTHONPATH="$ROOT/src" conda run -n x2-sonic-isaaclab python -m pytest -q \
  "$ROOT/tests/test_phase68_residual_ppo.py" \
  "$ROOT/tests/test_phase67_phase_conditioned_residual.py"

CWI_PHASE68_RESIDUAL_TRAIN=1 python3 "$LEDGER" \
  --label phase68_residual_one_optimizer_step \
  --resource-output "${REPORT}_train_resource.json" \
  --log "$LOGDIR/train.log" --disk-path "$ROOT" -- \
  conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
  --mode train --checkpoint "$ORIGINAL" --output "${REPORT}_train.json" \
  --source-output "$ARTIFACT/source_stage219_zero_residual.pt" \
  --final-output "$ARTIFACT/candidate_phase_conditioned_residual_one_step.pt" \
  --num-envs 64 --seed 42 --device cuda:0 --headless

for role in source candidate; do
  checkpoint="$ARTIFACT/source_stage219_zero_residual.pt"
  [[ "$role" == candidate ]] && checkpoint="$ARTIFACT/candidate_phase_conditioned_residual_one_step.pt"
  CWI_PHASE68_RESIDUAL_EVAL_ROLE="$role" python3 "$LEDGER" \
    --label "phase68_${role}_complete_event" \
    --resource-output "${REPORT}_${role}_resource.json" \
    --log "$LOGDIR/${role}_eval.log" --disk-path "$ROOT" -- \
    conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
    --mode eval --checkpoint "$ORIGINAL" --residual-checkpoint "$checkpoint" \
    --output "${REPORT}_${role}_eval.json" --num-envs 64 --seed 42 \
    --eval-steps 512 --device cuda:0 --headless
done

python3 "$FINALIZER" \
  --prereg "${REPORT}_prereg.json" \
  --train "${REPORT}_train.json" --train-resource "${REPORT}_train_resource.json" \
  --source "${REPORT}_source_eval.json" --source-resource "${REPORT}_source_resource.json" \
  --candidate "${REPORT}_candidate_eval.json" --candidate-resource "${REPORT}_candidate_resource.json" \
  --output "${REPORT}_result.json" --markdown "${REPORT}.md"
