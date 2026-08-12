#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
PYTHON=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
RUNNER="$ROOT/scripts/run_x2_phase77p_performance_campaign.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_phase77p_performance_campaign.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"
PREREG="$ROOT/reports/retarget/x2_phase77p_performance_campaign_prereg_v3.json"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
ART="$ROOT/artifacts/retarget/x2_phase77p_performance_campaign"
REPORT="$ROOT/reports/retarget"
LOG=/tmp/x2_phase77p_performance_logs
TRAIN_SEEDS=(770101 770102)
EVAL_SEEDS=(771001 771002 771003)
NUM_ENVS=256

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25
export CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20

sha() { sha256sum "$1" | awk '{print $1}'; }

verify_prereg() {
  "$PYTHON" - "$ROOT" "$PREREG" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, path = Path(sys.argv[1]), Path(sys.argv[2])
sidecar = path.with_suffix(path.suffix + ".sha256")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
assert sidecar.read_text() == f"{actual}  {path.name}\n"
d = json.loads(path.read_text())
assert d["schema"] == "x2_phase77p_performance_campaign_prereg_v1"
for section in ("immutable_code", "immutable_evidence"):
    for rel, expected in d[section].items():
        target = root / rel
        assert target.is_file(), rel
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, rel
for absolute, expected in d["immutable_inputs"].items():
    target = Path(absolute)
    assert target.is_file(), absolute
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, absolute
PY
}

gpu_idle_gate() {
  "$PYTHON" - <<'PY'
import json, subprocess, time
deadline = time.monotonic() + 180.0
rows = []
while time.monotonic() < deadline:
    compute = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    p=subprocess.run([
        "nvidia-smi", "--id=0",
        "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    row = [float(x.strip()) for x in p.stdout.strip().split(",")]
    passed = (
        not compute.stdout.strip()
        and row[0] <= 50.0
        and row[1] <= 1536.0
        and row[2] <= 80.0
        and row[3] <= 60.0
    )
    if passed:
        rows.append(row)
    else:
        rows.clear()
    if len(rows) == 10:
        print(json.dumps({"consecutive_idle_samples": rows}))
        raise SystemExit(0)
    time.sleep(1)
raise RuntimeError("GPU did not provide ten consecutive idle samples within 180 s")
PY
}

host_resource_gate() {
  local free avail
  free=$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')
  avail=$(awk '/MemAvailable/ {print $2*1024}' /proc/meminfo)
  [[ "$free" -ge 274877906944 ]]  # 256 GiB
  awk -v value="$avail" 'BEGIN { exit !(value >= 12884901888) }'
}

check_absent() {
  for path in "$@"; do
    [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
  done
}

sidecar() {
  local path=$1 target="${1}.sha256"
  [[ ! -e "$target" ]]
  printf '%s  %s\n' "$(sha "$path")" "$(basename "$path")" > "$target"
}

validate_resource() {
  local resource=$1 timeout=$2
  jq -e --argjson timeout "$timeout" '
    .exit_code == 0 and .raw_returncode == 0 and .autonomous_exit == true and
    .timed_out == false and .term_sent == false and .kill_sent == false and
    .forced_cleanup == false and .elapsed_s <= $timeout and
    .disk_used_delta_bytes <= 2147483648 and .disk_after.free_bytes >= 268435456000 and
    .gpu != null and .gpu.memory_used_peak_mib <= 20480
  ' "$resource" >/dev/null
}

cumulative_disk_gate() {
  local total
  total=$(jq -s '[.[].disk_used_delta_bytes | select(. > 0)] | add // 0' \
    "$REPORT"/x2_phase77p_*_resource.json)
  [[ "$total" -le 10737418240 ]]  # 10 GiB campaign hard cap
}

run_launch() {
  local label=$1 timeout=$2 report=$3
  shift 3
  local resource="$REPORT/x2_phase77p_${label}_resource.json"
  local log="$LOG/${label}.log"
  check_absent "$report" "${report}.sha256" "$resource" "${resource}.sha256" "$log" "${log}.sha256"
  gpu_idle_gate
  host_resource_gate
  set +e
  "$PYTHON" "$LEDGER" \
    --label "x2_phase77p_${label}" \
    --resource-output "$resource" \
    --log "$log" \
    --disk-path "$ROOT" \
    --timeout-seconds "$timeout" \
    --term-grace-seconds 5 \
    -- "$PYTHON" "$RUNNER" "$@" --report "$report" --device cuda:0 --headless
  local ledger_status=$?
  set -e
  [[ ! -e "$report" || -e "${report}.sha256" ]] || sidecar "$report"
  [[ "$ledger_status" -eq 0 ]] || return "$ledger_status"
  validate_resource "$resource" "$timeout"
  [[ -s "$report" ]]
  (cd "$(dirname "$resource")" && sha256sum -c "$(basename "${resource}.sha256")")
  (cd "$(dirname "$log")" && sha256sum -c "$(basename "${log}.sha256")")
  [[ -e "${report}.sha256" ]] || sidecar "$report"
  cumulative_disk_gate
}

train_segment() {
  local seed=$1 stage=$2
  local start resume_args=()
  if [[ "$stage" == 5 ]]; then
    start=0
  else
    start=5
    resume_args=(
      --resume "$ART/seed${seed}_u5.pt"
      --resume-report "$REPORT/x2_phase77p_seed${seed}_u5_train.json"
      --resume-gate "$REPORT/x2_phase77p_performance_u5_local_gate.json"
    )
  fi
  local checkpoint="$ART/seed${seed}_u${stage}.pt"
  local report="$REPORT/x2_phase77p_seed${seed}_u${stage}_train.json"
  check_absent "$checkpoint" "${checkpoint}.sha256"
  run_launch "seed${seed}_u${stage}_train" 600 "$report" \
    --mode train --source "$SOURCE" --prereg "$PREREG" \
    --num-envs "$NUM_ENVS" --seed "$seed" \
    --start-update "$start" --updates 5 --steps-per-env 48 \
    "${resume_args[@]}" --checkpoint-output "$checkpoint"
  jq -e --argjson stage "$stage" '
    .schema == "x2_phase77p_performance_train_v1" and
    .decision == "SEGMENT_VALID" and .end_update == $stage and
    .optimizer_steps == 20
  ' "$report" >/dev/null
  sidecar "$checkpoint"
}

validate_completed_train() {
  local seed=$1 stage=$2
  local checkpoint="$ART/seed${seed}_u${stage}.pt"
  local report="$REPORT/x2_phase77p_seed${seed}_u${stage}_train.json"
  local resource="$REPORT/x2_phase77p_seed${seed}_u${stage}_train_resource.json"
  local log="$LOG/seed${seed}_u${stage}_train.log"
  for path in "$checkpoint" "$report" "$resource" "$log"; do
    [[ -s "$path" && -s "${path}.sha256" ]]
    (cd "$(dirname "$path")" && sha256sum -c "$(basename "${path}.sha256")")
  done
  validate_resource "$resource" 600
  jq -e --argjson seed "$seed" --argjson stage "$stage" --arg checkpoint_sha "$(sha "$checkpoint")" '
    .schema == "x2_phase77p_performance_train_v1" and
    .decision == "SEGMENT_VALID" and
    .seed == $seed and .start_update == 0 and .end_update == $stage and
    .optimizer_steps == 20 and .checkpoint_sha256 == $checkpoint_sha and
    ([.updates[].gates | to_entries[].value] | all)
  ' "$report" >/dev/null
}

eval_stage() {
  local train_seed=$1 eval_seed=$2 stage=$3 lane=$4
  local checkpoint="$ART/seed${train_seed}_u${stage}.pt"
  local report="$REPORT/x2_phase77p_seed${train_seed}_u${stage}_eval${eval_seed}_lane${lane}.json"
  run_launch "seed${train_seed}_u${stage}_eval${eval_seed}_lane${lane}" 300 "$report" \
    --mode eval --source "$SOURCE" --prereg "$PREREG" \
    --candidate "$checkpoint" --lane "$lane" \
    --num-envs 128 --seed "$eval_seed" --eval-steps 512
  jq -e '
    .schema == "x2_phase77p_performance_eval_v1" and
    .decision == "EVAL_FINITE" and .finite == true and
    .optimizer_steps == 0 and .checkpoint_writes == 0
  ' "$report" >/dev/null
}

finalize_stage() {
  local stage=$1 output="$REPORT/x2_phase77p_performance_u${1}_local_gate.json"
  local command=("$PYTHON" "$FINALIZER" --stage "$stage")
  check_absent "$output" "${output}.sha256"
  for seed in "${TRAIN_SEEDS[@]}"; do
    command+=(--train "$REPORT/x2_phase77p_seed${seed}_u${stage}_train.json")
  done
  for seed in "${TRAIN_SEEDS[@]}"; do
    for eval_seed in "${EVAL_SEEDS[@]}"; do
      for lane in A B; do
        command+=(--eval "$REPORT/x2_phase77p_seed${seed}_u${stage}_eval${eval_seed}_lane${lane}.json")
      done
    done
  done
  if [[ "$stage" == 10 ]]; then
    for seed in "${TRAIN_SEEDS[@]}"; do
      command+=(--stage5-train "$REPORT/x2_phase77p_seed${seed}_u5_train.json")
    done
    command+=(--stage5-gate "$REPORT/x2_phase77p_performance_u5_local_gate.json")
  fi
  command+=(--output "$output")
  "${command[@]}"
}

verify_prereg
host_resource_gate
gpu_idle_gate
mkdir -p "$ART" "$REPORT" "$LOG"

# Both independent sources must show a signal at update 5 before either is continued.
for seed in "${TRAIN_SEEDS[@]}"; do
  if [[ -e "$ART/seed${seed}_u5.pt" || -e "$REPORT/x2_phase77p_seed${seed}_u5_train.json" ]]; then
    validate_completed_train "$seed" 5
  else
    train_segment "$seed" 5
  fi
done
for seed in "${TRAIN_SEEDS[@]}"; do
  for eval_seed in "${EVAL_SEEDS[@]}"; do
    for lane in A B; do eval_stage "$seed" "$eval_seed" 5 "$lane"; done
  done
done
finalize_stage 5

for seed in "${TRAIN_SEEDS[@]}"; do train_segment "$seed" 10; done
for seed in "${TRAIN_SEEDS[@]}"; do
  for eval_seed in "${EVAL_SEEDS[@]}"; do
    for lane in A B; do eval_stage "$seed" "$eval_seed" 10 "$lane"; done
  done
done
finalize_stage 10

echo "Phase77p local pilot completed; official panel remains a separate fail-closed gate."
