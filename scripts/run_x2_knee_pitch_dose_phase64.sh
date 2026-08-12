#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
REPORTDIR="$ROOT/reports/retarget"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_knee_pitch_dose_phase64.py"
PREREG="$REPORTDIR/x2_knee_pitch_dose_phase64_prereg.json"
LOGDIR=/tmp/x2_knee_pitch_dose_phase64_logs

SCREEN="$REPORTDIR/x2_knee_pitch_dose_phase64_screen.json"
RESOURCE="$REPORTDIR/x2_knee_pitch_dose_phase64_resource.json"
RESULT="$REPORTDIR/x2_knee_pitch_dose_phase64_result.json"
MARKDOWN="$REPORTDIR/x2_knee_pitch_dose_phase64.md"

for path in "$SCREEN" "$RESOURCE" "$RESULT" "$MARKDOWN"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite immutable Phase64 output: $path" >&2; exit 2; }
done
[[ -f "$PREREG" ]] || { echo "missing Phase64 preregistration: $PREREG" >&2; exit 2; }
free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 20 * 1024 * 1024 )) || {
  echo "Phase64 requires at least 20 GiB free before launch" >&2
  exit 2
}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=1.0
export CWI_UPPER_DETERMINISTIC_SPLIT=0 CWI_UPPER_SPLIT_MODE=contiguous
export CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
export CWI_PHASE64_KNEE_DOSE_SCREEN=1
mkdir -p "$REPORTDIR" "$LOGDIR"

if ! python3 "$LEDGER" \
  --label phase64_knee_pitch_dose_screen \
  --resource-output "$RESOURCE" \
  --log "$LOGDIR/screen.log" \
  --disk-path "$ROOT" -- \
  conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
  --mode screen --checkpoint "$ORIGINAL" --output "$SCREEN" \
  --num-envs 64 --seed 42 --eval-steps 200 --device cuda:0 --headless; then
  tail -n 160 "$LOGDIR/screen.log" >&2
  exit 1
fi

python3 "$FINALIZER" \
  --input "$SCREEN" --resource "$RESOURCE" --prereg "$PREREG" \
  --output "$RESULT" --markdown "$MARKDOWN"

python3 - "$RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"Phase64 decision={result['decision']}")
print(f"selected={result['selected_group']}")
PY
