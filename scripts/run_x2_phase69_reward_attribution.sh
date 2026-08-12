#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
PHASE68_ARTIFACT="$ROOT/artifacts/retarget/x2_phase_conditioned_residual_phase68"
REPORT="$ROOT/reports/retarget/x2_phase69_reward_attribution"
EVIDENCE="$ROOT/artifacts/retarget/x2_phase69_reward_attribution/rollout_evidence.pt"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_phase69_reward_attribution.py"
LOG=/tmp/x2_phase69_logs/attribution.log

outputs=(
  "${REPORT}_screen.json"
  "${REPORT}_resource.json"
  "${REPORT}_result.json"
  "${REPORT}.md"
  "$EVIDENCE"
)
for path in "${outputs[@]}"; do
  [[ ! -e "$path" ]] || {
    echo "refusing to overwrite immutable Phase69 output: $path" >&2
    exit 2
  }
done
free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 20 * 1024 * 1024 )) || {
  echo "Phase69 requires at least 20 GiB free" >&2
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

PYTHONPATH="$ROOT/src:$ROOT" conda run -n x2-sonic-isaaclab python -m pytest -q \
  "$ROOT/tests/test_phase69_reward_attribution.py" \
  "$ROOT/tests/test_phase68_residual_ppo.py" \
  "$ROOT/tests/test_phase67_phase_conditioned_residual.py"

CWI_PHASE69_REWARD_ATTRIBUTION=1 python3 "$LEDGER" \
  --label phase69_zero_optimizer_reward_gradient_attribution \
  --resource-output "${REPORT}_resource.json" \
  --log "$LOG" --disk-path "$ROOT" -- \
  conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
  --mode train --checkpoint "$ORIGINAL" \
  --residual-checkpoint "$PHASE68_ARTIFACT/source_stage219_zero_residual.pt" \
  --attribution-candidate-checkpoint "$PHASE68_ARTIFACT/candidate_phase_conditioned_residual_one_step.pt" \
  --rollout-bundle "$EVIDENCE" --output "${REPORT}_screen.json" \
  --num-envs 64 --seed 42 --device cuda:0 --headless

python3 "$FINALIZER" \
  --prereg "${REPORT}_prereg.json" \
  --screen "${REPORT}_screen.json" \
  --resource "${REPORT}_resource.json" \
  --output "${REPORT}_result.json" \
  --markdown "${REPORT}.md"
