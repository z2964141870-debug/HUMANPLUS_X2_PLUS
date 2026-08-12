#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
OUTDIR="$ROOT/artifacts/retarget/x2_native_transition_posture_phase61"
REPORTDIR="$ROOT/reports/retarget"
RUNNER="$ROOT/scripts/run_x2_upper_robust_one_update_phase56.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_native_transition_posture_phase61.py"
LOGDIR=/tmp/x2_native_transition_posture_phase61_logs

SOURCE_EVAL="$REPORTDIR/x2_native_transition_posture_phase61_source_eval.json"
TRAIN_REPORT="$REPORTDIR/x2_native_transition_posture_phase61_train.json"
CANDIDATE_EVAL="$REPORTDIR/x2_native_transition_posture_phase61_candidate_eval.json"
SOURCE_WEIGHT="$OUTDIR/source_stage219_zero_lora.pt"
CANDIDATE_WEIGHT="$OUTDIR/candidate_joint_transition_one_update.pt"
RESULT="$REPORTDIR/x2_native_transition_posture_phase61_result.json"
MARKDOWN="$REPORTDIR/x2_native_transition_posture_phase61.md"

for path in \
  "$SOURCE_EVAL" "$TRAIN_REPORT" "$CANDIDATE_EVAL" \
  "$SOURCE_WEIGHT" "$CANDIDATE_WEIGHT" "$RESULT" "$MARKDOWN" \
  "$REPORTDIR/x2_native_transition_posture_phase61_source_resource.json" \
  "$REPORTDIR/x2_native_transition_posture_phase61_train_resource.json" \
  "$REPORTDIR/x2_native_transition_posture_phase61_candidate_resource.json"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite immutable Phase61 output: $path" >&2; exit 2; }
done

free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 20 * 1024 * 1024 )) || {
  echo "Phase61 requires at least 20 GiB free before launch" >&2
  exit 2
}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=1.0
export CWI_UPPER_DETERMINISTIC_SPLIT=0 CWI_UPPER_SPLIT_MODE=contiguous
export CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20
export CWI_PHASE61_TRANSITION_POSTURE=1
mkdir -p "$OUTDIR" "$REPORTDIR" "$LOGDIR"

run_ledger() {
  local label=$1 resource=$2 log=$3
  shift 3
  if ! python3 "$LEDGER" \
    --label "$label" --resource-output "$resource" --log "$log" \
    --disk-path "$ROOT" -- "$@"; then
    tail -n 160 "$log" >&2
    return 1
  fi
}

COMMON=(--num-envs 64 --seed 42 --eval-steps 512 --device cuda:0 --headless)

run_ledger phase61_source_eval \
  "$REPORTDIR/x2_native_transition_posture_phase61_source_resource.json" \
  "$LOGDIR/source_eval.log" \
  conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
  --mode eval --checkpoint "$ORIGINAL" --output "$SOURCE_EVAL" "${COMMON[@]}"

run_ledger phase61_train \
  "$REPORTDIR/x2_native_transition_posture_phase61_train_resource.json" \
  "$LOGDIR/train.log" \
  conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
  --mode train --checkpoint "$ORIGINAL" --output "$TRAIN_REPORT" \
  --source-output "$SOURCE_WEIGHT" --final-output "$CANDIDATE_WEIGHT" \
  "${COMMON[@]}"

run_ledger phase61_candidate_eval \
  "$REPORTDIR/x2_native_transition_posture_phase61_candidate_resource.json" \
  "$LOGDIR/candidate_eval.log" \
  conda run --no-capture-output -n x2-sonic-isaaclab python "$RUNNER" \
  --mode eval --checkpoint "$CANDIDATE_WEIGHT" --output "$CANDIDATE_EVAL" \
  "${COMMON[@]}"

python3 "$FINALIZER" --root "$ROOT" --output "$RESULT" --markdown "$MARKDOWN"

python3 - "$RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"Phase61 decision={result['decision']}")
print(f"failed={[name for name, passed in result['checks'].items() if not passed]}")
PY
