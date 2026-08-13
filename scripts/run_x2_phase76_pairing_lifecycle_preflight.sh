#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
PYTHON=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
ORIGINAL="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
SOURCE="$ROOT/artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt"
PREREG="$ROOT/reports/retarget/x2_phase76_pairing_lifecycle_preflight_prereg.json"
RUNNER="$ROOT/scripts/run_x2_phase76_pairing_lifecycle_preflight.py"
SUPERVISOR="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_phase76_pairing_lifecycle_preflight.py"
ART_DIR="$ROOT/artifacts/retarget/x2_phase76_pairing_lifecycle_preflight"
REPORT_DIR="$ROOT/reports/retarget"
LOG_DIR=/tmp/x2_phase76_logs
FINAL_RESULT="$REPORT_DIR/x2_phase76_pairing_lifecycle_preflight_result.json"
FINAL_MD="$REPORT_DIR/x2_phase76_pairing_lifecycle_preflight.md"

mkdir -p "$ART_DIR" "$LOG_DIR"
outputs=("$FINAL_RESULT" "${FINAL_RESULT}.sha256" "$FINAL_MD")
for seed in 0 1 2; do
  outputs+=(
    "$ART_DIR/seed${seed}_commit.json" "$ART_DIR/seed${seed}_commit.json.sha256"
    "$REPORT_DIR/x2_phase76_seed${seed}_screen.json" "$REPORT_DIR/x2_phase76_seed${seed}_screen.json.sha256"
    "$REPORT_DIR/x2_phase76_seed${seed}_failure.json" "$REPORT_DIR/x2_phase76_seed${seed}_failure.json.sha256"
    "$REPORT_DIR/x2_phase76_seed${seed}_resource.json" "$REPORT_DIR/x2_phase76_seed${seed}_resource.json.sha256"
    "$LOG_DIR/seed${seed}.log" "$LOG_DIR/seed${seed}.log.sha256"
  )
done
for path in "${outputs[@]}"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite Phase76 output: $path" >&2; exit 2; }
done
(cd "$(dirname "$PREREG")" && sha256sum -c "$(basename "$PREREG").sha256" >/dev/null)
free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
(( free_kib >= 32 * 1024 * 1024 )) || { echo "Phase76 requires 32 GiB free" >&2; exit 2; }
FREEZE="$REPORT_DIR/x2_phase75_seed0_resource_failure_freeze.json"
BACKUP_MANIFEST="$REPORT_DIR/x2_phase75_seed0_resource_failure_backup_manifest.json"
[[ "$(sha256sum "$FREEZE" | awk '{print $1}')" == "$(jq -er '.phase75_disclosure.freeze_sha256' "$PREREG")" ]] || {
  echo "Phase76 Phase75 failure freeze drifted" >&2; exit 2;
}
[[ "$(sha256sum "$BACKUP_MANIFEST" | awk '{print $1}')" == "$(jq -er '.phase75_disclosure.backup_manifest_sha256' "$PREREG")" ]] || {
  echo "Phase76 Phase75 backup manifest drifted" >&2; exit 2;
}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$ROOT/scripts:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20

PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON" -m pytest -q \
  "$ROOT/tests/test_phase76_pairing_lifecycle_preflight.py" \
  "$ROOT/tests/test_phase76_pairing_lifecycle_finalizer.py" \
  "$ROOT/tests/test_phase76_deadline_supervisor.py"

env_seeds=(760041 760042 760043)
screen_args=()
resource_args=()
cumulative_disk_delta=0
seen_source_initial_hashes=()
for seed in 0 1 2; do
  commit="$ART_DIR/seed${seed}_commit.json"
  screen="$REPORT_DIR/x2_phase76_seed${seed}_screen.json"
  failure="$REPORT_DIR/x2_phase76_seed${seed}_failure.json"
  resource="$REPORT_DIR/x2_phase76_seed${seed}_resource.json"
  log="$LOG_DIR/seed${seed}.log"
  set +e
  "$PYTHON" "$SUPERVISOR" \
    --label "phase76_seed${seed}_single_process_pairing_preflight" \
    --resource-output "$resource" --log "$log" --disk-path "$ROOT" \
    --timeout-seconds 110 --term-grace-seconds 5 -- \
    "$PYTHON" "$RUNNER" \
      --checkpoint "$ORIGINAL" --residual-checkpoint "$SOURCE" --prereg "$PREREG" \
      --seed-index "$seed" --env-seed "${env_seeds[$seed]}" \
      --commit "$commit" --output "$screen" --failure "$failure" \
      --num-envs 128 --device cuda:0 --headless
  status=$?
  set -e
  if [[ "$status" -ne 0 || ! -f "$screen" || ! -f "$resource" ]]; then
    echo "Phase76 seed=$seed failed (status=$status, screen=$([[ -f "$screen" ]] && echo present || echo absent))" >&2
    exit 1
  fi
  (cd "$(dirname "$resource")" && sha256sum -c "$(basename "$resource").sha256" >/dev/null)
  (cd "$(dirname "$log")" && sha256sum -c "$(basename "$log").sha256" >/dev/null)
  jq -e '.decision == "PASS_INITIAL_PAIRING_LAUNCH"
    and .all_pairs_pass == true
    and .unknown_mutable_tensor_fields_empty == true
    and .mixed_device_evidence_path_exercised == true
    and (.cpu_manifest_fields | length) > 0
    and .source_donor_initial.nondegenerate == true
    and .observation_recomputed_after_clone == true
    and .physx_low_level_readback_present == true
    and .scientific_metrics_present == false' "$screen" >/dev/null
  jq -e '.schema == "x2_gpu_deadline_ledger_phase76_v1"
    and .exit_code == 0 and .raw_returncode == 0 and .autonomous_exit == true
    and .timed_out == false and .term_sent == false and .kill_sent == false
    and .forced_cleanup == false and .gpu != null
    and .gpu.memory_used_peak_mib <= 8192 and .elapsed_s <= 120' "$resource" >/dev/null
  source_initial_hash=$(jq -er '.source_donor_initial.combined_sha256' "$screen")
  for prior_hash in "${seen_source_initial_hashes[@]:-}"; do
    [[ "$source_initial_hash" != "$prior_hash" ]] || { echo "Phase76 duplicate donor initial hashes" >&2; exit 1; }
  done
  seen_source_initial_hashes+=("$source_initial_hash")
  delta=$(jq -r '.disk_used_delta_bytes' "$resource")
  (( delta <= 536870912 )) || { echo "Phase76 launch disk delta exceeded 512 MiB" >&2; exit 1; }
  positive=$delta; (( positive >= 0 )) || positive=0
  cumulative_disk_delta=$((cumulative_disk_delta + positive))
  (( cumulative_disk_delta <= 1610612736 )) || { echo "Phase76 cumulative disk delta exceeded 1.5 GiB" >&2; exit 1; }
  free_after=$(jq -r '.disk_after.free_bytes' "$resource")
  (( free_after >= 28 * 1024 * 1024 * 1024 )) || { echo "Phase76 free disk below 28 GiB" >&2; exit 1; }
  (( $(stat -c %s "$commit") + $(stat -c %s "$screen") <= 16777216 )) || {
    echo "Phase76 technical evidence exceeded 16 MiB" >&2; exit 1;
  }
  screen_args+=(--screen "$screen")
  resource_args+=(--resource "$resource")
done

"$PYTHON" "$FINALIZER" --prereg "$PREREG" \
  "${screen_args[@]}" "${resource_args[@]}" \
  --output "$FINAL_RESULT" --markdown "$FINAL_MD"
