#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
OUTDIR="$ROOT/artifacts/retarget/x2_native_posture_phase60"
REPORTDIR="$ROOT/reports/retarget"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LOGDIR=/tmp/x2_native_posture_phase60_logs
COMMON=(--num-envs 64 --device cuda:0 --headless)

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=1.0
export CWI_UPPER_DETERMINISTIC_SPLIT=0 CWI_UPPER_SPLIT_MODE=contiguous
export CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
mkdir -p "$OUTDIR" "$REPORTDIR" "$LOGDIR"

run_eval() {
  local group=$1 checkpoint=$2 seed=$3 output=$4
  local log="$LOGDIR/${group}_seed${seed}_eval.log"
  if ! CWI_PHASE60_POSTURE_VARIANT="$group" \
    conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
      --mode eval --checkpoint "$checkpoint" --eval-steps 200 --seed "$seed" \
      --output "$output" "${COMMON[@]}" >"$log" 2>&1; then
    tail -n 120 "$log" >&2
    return 1
  fi
  test -s "$output"
  python3 - "$output" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); m=d["groups"]["all"]
print(f"eval={d['posture_variant']}/seed{d['seed']} pitch={m['signed_pitch_rad']['mean']:.6f} "
      f"speed_rmse={m['velocity_tracking_rmse']:.6f} survival={m['survival_s_mean']:.3f}")
PY
}

run_train() {
  local group=$1 output=$2 final=$3
  local log="$LOGDIR/${group}_train.log"
  if ! CWI_PHASE60_POSTURE_VARIANT="$group" \
    conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
      --mode train --checkpoint "$ORIGINAL" --seed 42 \
      --source-output "$OUTDIR/source_stage219_zero_lora.pt" \
      --final-output "$final" --output "$output" "${COMMON[@]}" >"$log" 2>&1; then
    tail -n 120 "$log" >&2
    return 1
  fi
  test -s "$output"
  test -s "$final"
  python3 - "$output" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); k=d["fixed_source_retention"]
print(f"train={d['posture_variant']} optimizer_steps={d['optimizer_steps']} "
      f"kl_mean={k['kl_mean']:.8f} action_drift={k['action_max_abs']:.8f}")
PY
}

run_eval A "$ORIGINAL" 40 "$REPORTDIR/x2_native_posture_phase60_A_seed40_eval.json"
run_eval A "$ORIGINAL" 41 "$REPORTDIR/x2_native_posture_phase60_A_seed41_eval.json"
if [[ ! -s "$REPORTDIR/x2_native_posture_phase60_A_seed42_smoke.json" ]]; then
  run_eval A "$ORIGINAL" 42 "$REPORTDIR/x2_native_posture_phase60_A_seed42_smoke.json"
fi

run_train B "$REPORTDIR/x2_native_posture_phase60_B_train.json" "$OUTDIR/candidate_B_one_update.pt"
for seed in 40 41 42; do
  run_eval B "$OUTDIR/candidate_B_one_update.pt" "$seed" \
    "$REPORTDIR/x2_native_posture_phase60_B_seed${seed}_eval.json"
done

run_train C "$REPORTDIR/x2_native_posture_phase60_C_train.json" "$OUTDIR/candidate_C_one_update.pt"
for seed in 40 41 42; do
  run_eval C "$OUTDIR/candidate_C_one_update.pt" "$seed" \
    "$REPORTDIR/x2_native_posture_phase60_C_seed${seed}_eval.json"
done

PYTHONPATH="$ROOT/src" conda run -n x2-sonic-isaaclab python \
  "$ROOT/tools/retarget/finalize_x2_native_posture_phase60.py" \
  --root "$ROOT" --output "$REPORTDIR/x2_native_posture_phase60_result.json"
