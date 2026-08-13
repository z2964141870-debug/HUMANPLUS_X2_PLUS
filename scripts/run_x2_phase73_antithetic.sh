#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
PYTHON=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
SOURCE="$ROOT/artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt"
PREREG="$ROOT/reports/retarget/x2_phase73_antithetic_prereg.json"
RUNNER="$ROOT/scripts/run_x2_phase73_antithetic.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_ledger.py"
PAIR_VALIDATOR="$ROOT/tools/retarget/validate_x2_phase73_pair.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_phase73_antithetic.py"
SCHEDULE_DIR="$ROOT/artifacts/retarget/x2_phase73_schedules"
ART_DIR="$ROOT/artifacts/retarget/x2_phase73_antithetic"
REPORT_DIR="$ROOT/reports/retarget"
LOG_DIR=/tmp/x2_phase73_logs
FINAL_RESULT="$REPORT_DIR/x2_phase73_antithetic_result.json"
FINAL_MD="$REPORT_DIR/x2_phase73_antithetic.md"

mkdir -p "$ART_DIR" "$LOG_DIR"

outputs=("$FINAL_RESULT" "${FINAL_RESULT}.sha256" "$FINAL_MD")
for seed in 0 1 2 3 4; do
  outputs+=(
    "$ART_DIR/seed${seed}_initial_commit.json"
    "$ART_DIR/seed${seed}_initial_commit.json.sha256"
    "$ART_DIR/seed${seed}_pair.pt"
    "$ART_DIR/seed${seed}_pair.pt.sha256"
    "$REPORT_DIR/x2_phase73_seed${seed}_pair_result.json"
    "$REPORT_DIR/x2_phase73_seed${seed}_pair_result.json.sha256"
  )
  for sign in plus minus; do
    outputs+=(
      "$ART_DIR/seed${seed}_${sign}_rollout.pt"
      "$ART_DIR/seed${seed}_${sign}_rollout.pt.sha256"
      "$REPORT_DIR/x2_phase73_seed${seed}_${sign}_screen.json"
      "$REPORT_DIR/x2_phase73_seed${seed}_${sign}_screen.json.sha256"
      "$REPORT_DIR/x2_phase73_seed${seed}_${sign}_resource.json"
      "$REPORT_DIR/x2_phase73_seed${seed}_${sign}_resource.json.sha256"
      "$LOG_DIR/seed${seed}_${sign}.log"
      "$LOG_DIR/seed${seed}_${sign}.log.sha256"
    )
  done
done
for path in "${outputs[@]}"; do
  [[ ! -e "$path" ]] || {
    echo "refusing to overwrite immutable Phase73 output: $path" >&2
    exit 2
  }
done
free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 32 * 1024 * 1024 )) || {
  echo "Phase73 requires at least 32 GiB free" >&2
  exit 2
}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$ROOT/scripts:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25
export CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" -m pytest -q \
  "$ROOT/tests/test_phase73_antithetic.py" \
  "$ROOT/tests/test_phase73_antithetic_runner.py" \
  "$ROOT/tests/test_phase73_antithetic_finalizer.py" \
  "$ROOT/tests/test_phase72_antithetic.py" \
  "$ROOT/tests/test_phase72_antithetic_runner.py" \
  "$ROOT/tests/test_phase72_antithetic_finalizer.py" \
  "$ROOT/tests/test_phase70_long_lookahead.py" \
  "$ROOT/tests/test_phase68_residual_ppo.py" \
  "$ROOT/tests/test_phase67_phase_conditioned_residual.py"

env_seeds=(730041 730042 730043 730044 730045)
resource_args=()
pair_result_args=()
cumulative_disk_delta=0
declare -A seen_initial_hashes=()

write_sidecar() {
  local path=$1
  local directory basename
  directory=$(dirname "$path")
  basename=$(basename "$path")
  (cd "$directory" && sha256sum "$basename" > "${basename}.sha256")
}

run_launch() {
  local seed_index=$1 sign=$2 init_mode=$3
  local screen="$REPORT_DIR/x2_phase73_seed${seed_index}_${sign}_screen.json"
  local resource="$REPORT_DIR/x2_phase73_seed${seed_index}_${sign}_resource.json"
  local bundle="$ART_DIR/seed${seed_index}_${sign}_rollout.pt"
  local initial="$ART_DIR/seed${seed_index}_initial_commit.json"
  local schedule="$SCHEDULE_DIR/epsilon_seed${seed_index}.pt"
  local log="$LOG_DIR/seed${seed_index}_${sign}.log"
  set +e
  "$PYTHON" "$LEDGER" \
    --label "phase73_seed${seed_index}_${sign}_antithetic_zero_optimizer" \
    --resource-output "$resource" --log "$log" --disk-path "$ROOT" -- \
    "$PYTHON" "$RUNNER" \
      --checkpoint "$ORIGINAL" --residual-checkpoint "$SOURCE" \
      --schedule "$schedule" --prereg "$PREREG" \
      --seed-index "$seed_index" --env-seed "${env_seeds[$seed_index]}" \
      --sign "$sign" --init-mode "$init_mode" --init-commit "$initial" \
      --bundle "$bundle" --output "$screen" \
      --num-envs 64 --steps 400 --device cuda:0 --headless
  local status=$?
  set -e
  [[ -f "$resource" ]] && write_sidecar "$resource"
  [[ -f "$log" ]] && write_sidecar "$log"
  if [[ "$status" -ne 0 || ! -f "$screen" ]]; then
    echo "Phase73 launch seed=$seed_index sign=$sign failed (status=$status)" >&2
    exit 1
  fi
  jq -e '.schema == "x2_phase73_antithetic_screen_v1" and .decision == "VALID_PENDING_PAIR"' "$screen" >/dev/null
  jq -e '.exit_code == 0' "$resource" >/dev/null
  local delta positive_delta free_after gpu_peak bundle_bytes
  delta=$(jq -r '.disk_used_delta_bytes' "$resource")
  positive_delta=$delta
  (( positive_delta >= 0 )) || positive_delta=0
  cumulative_disk_delta=$(( cumulative_disk_delta + positive_delta ))
  (( delta <= 536870912 )) || {
    echo "Phase73 launch exceeded 512 MiB hard disk delta" >&2
    exit 1
  }
  gpu_peak=$(jq -r 'if .gpu == null then "null" else .gpu.memory_used_peak_mib end' "$resource")
  [[ "$gpu_peak" != "null" ]] && awk -v value="$gpu_peak" 'BEGIN { exit !(value <= 5120) }' || {
    echo "Phase73 launch has no valid GPU ledger sample or exceeded 5120 MiB" >&2
    exit 1
  }
  bundle_bytes=$(jq -r '.bundle_bytes' "$screen")
  (( bundle_bytes <= 33554432 )) || {
    echo "Phase73 raw bundle exceeded 32 MiB" >&2
    exit 1
  }
  free_after=$(jq -r '.disk_after.free_bytes' "$resource")
  (( free_after >= 28 * 1024 * 1024 * 1024 )) || {
    echo "Phase73 free space fell below 28 GiB" >&2
    exit 1
  }
  (( cumulative_disk_delta <= 4294967296 )) || {
    echo "Phase73 cumulative disk delta exceeded 4 GiB" >&2
    exit 1
  }
  if (( delta > 268435456 )); then
    echo "Phase73 per-launch 256 MiB soft review: hard resource gates remain satisfied" >&2
  fi
  if (( cumulative_disk_delta > 2147483648 )); then
    echo "Phase73 cumulative 2 GiB soft review: hard resource gates remain satisfied" >&2
  fi
  resource_args+=(--resource "$resource")
}

for seed in 0 1 2 3 4; do
  if (( seed % 2 == 0 )); then
    first=plus; second=minus
  else
    first=minus; second=plus
  fi
  run_launch "$seed" "$first" write
  initial_hash=$(jq -r '.combined_sha256' "$ART_DIR/seed${seed}_initial_commit.json")
  [[ -n "$initial_hash" && "$initial_hash" != "null" ]] || {
    echo "Phase73 seed $seed initial commit is invalid" >&2
    exit 1
  }
  if [[ -n "${seen_initial_hashes[$initial_hash]:-}" ]]; then
    echo "Phase73 seed $seed repeated an earlier initial state; stopping before pair completion" >&2
    exit 1
  fi
  seen_initial_hashes[$initial_hash]=$seed
  run_launch "$seed" "$second" verify
  pair_result="$REPORT_DIR/x2_phase73_seed${seed}_pair_result.json"
  "$PYTHON" "$PAIR_VALIDATOR" \
    --prereg "$PREREG" \
    --plus-screen "$REPORT_DIR/x2_phase73_seed${seed}_plus_screen.json" \
    --minus-screen "$REPORT_DIR/x2_phase73_seed${seed}_minus_screen.json" \
    --plus-resource "$REPORT_DIR/x2_phase73_seed${seed}_plus_resource.json" \
    --minus-resource "$REPORT_DIR/x2_phase73_seed${seed}_minus_resource.json" \
    --pair-evidence "$ART_DIR/seed${seed}_pair.pt" \
    --output "$pair_result"
  jq -e '.decision == "PAIR_VALID_PENDING_FINAL"' "$pair_result" >/dev/null
  pair_result_args+=(--pair-result "$pair_result")
done

"$PYTHON" "$FINALIZER" \
  --prereg "$PREREG" \
  "${pair_result_args[@]}" \
  "${resource_args[@]}" \
  --output "$FINAL_RESULT" --markdown "$FINAL_MD"
