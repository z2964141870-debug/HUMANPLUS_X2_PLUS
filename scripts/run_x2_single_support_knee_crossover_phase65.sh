#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
REPORTDIR="$ROOT/reports/retarget"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_single_support_knee_crossover_phase65.py"
PREREG="$REPORTDIR/x2_single_support_knee_crossover_phase65_prereg.json"
LOGDIR=/tmp/x2_single_support_knee_crossover_phase65_logs

PASS0="$REPORTDIR/x2_single_support_knee_crossover_phase65_pass0.json"
PASS1="$REPORTDIR/x2_single_support_knee_crossover_phase65_pass1.json"
RESOURCE0="$REPORTDIR/x2_single_support_knee_crossover_phase65_pass0_resource.json"
RESOURCE1="$REPORTDIR/x2_single_support_knee_crossover_phase65_pass1_resource.json"
RESULT="$REPORTDIR/x2_single_support_knee_crossover_phase65_result.json"
MARKDOWN="$REPORTDIR/x2_single_support_knee_crossover_phase65.md"

for path in "$PASS0" "$PASS1" "$RESOURCE0" "$RESOURCE1" "$RESULT" "$MARKDOWN"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite immutable Phase65 output: $path" >&2; exit 2; }
done
[[ -f "$PREREG" ]] || { echo "missing Phase65 preregistration: $PREREG" >&2; exit 2; }

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=1.0
export CWI_UPPER_DETERMINISTIC_SPLIT=0 CWI_UPPER_SPLIT_MODE=contiguous
export CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
mkdir -p "$REPORTDIR" "$LOGDIR"

for pass_index in 0 1; do
  free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
  (( free_kib >= 20 * 1024 * 1024 )) || {
    echo "Phase65 requires at least 20 GiB free before pass ${pass_index}" >&2
    exit 2
  }
  export CWI_PHASE65_CROSSOVER_PASS="$pass_index"
  screen_var="PASS${pass_index}"
  resource_var="RESOURCE${pass_index}"
  screen="${!screen_var}"
  resource="${!resource_var}"
  if ! python3 "$LEDGER" \
    --label "phase65_single_support_knee_crossover_pass${pass_index}" \
    --resource-output "$resource" \
    --log "$LOGDIR/pass${pass_index}.log" \
    --disk-path "$ROOT" -- \
    conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
    --mode screen --checkpoint "$ORIGINAL" --output "$screen" \
    --num-envs 64 --seed 42 --eval-steps 200 --device cuda:0 --headless; then
    tail -n 160 "$LOGDIR/pass${pass_index}.log" >&2
    exit 1
  fi
done
unset CWI_PHASE65_CROSSOVER_PASS

python3 "$FINALIZER" \
  --pass0 "$PASS0" --pass1 "$PASS1" \
  --resource0 "$RESOURCE0" --resource1 "$RESOURCE1" \
  --prereg "$PREREG" --output "$RESULT" --markdown "$MARKDOWN"

python3 - "$RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"Phase65 decision={result['decision']}")
print(f"pairing={result['pairing']['passed']}")
PY
