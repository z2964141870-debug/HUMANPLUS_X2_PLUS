#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
PYTHON=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
RUNNER="$ROOT/scripts/run_x2_privileged_teacher_reachability_v2.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"
PREREG="$ROOT/reports/retarget/x2_privileged_teacher_reachability_v2b_prereg.json"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
REPORT="$ROOT/reports/retarget/x2_privileged_teacher_reachability_v2b_result.json"
FAILURE="$ROOT/reports/retarget/x2_privileged_teacher_reachability_v2b_failure.json"
RESOURCE="$ROOT/reports/retarget/x2_privileged_teacher_reachability_v2b_resource.json"
LOG=/tmp/x2_privileged_teacher_reachability_v2b.log

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25
export CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20

for path in "$REPORT" "${REPORT}.sha256" "$FAILURE" "${FAILURE}.sha256" \
  "$RESOURCE" "${RESOURCE}.sha256" "$LOG" "${LOG}.sha256"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

(cd "$(dirname "$PREREG")" && sha256sum -c "$(basename "${PREREG}.sha256")")
free_bytes=$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')
[[ "$free_bytes" -ge 34359738368 ]]

"$PYTHON" - <<'PY'
import subprocess, time
deadline=time.monotonic()+180
good=0
while time.monotonic()<deadline:
    compute=subprocess.run([
        "nvidia-smi","--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
    row=subprocess.run([
        "nvidia-smi","--id=0","--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
    values=[float(x.strip()) for x in row.stdout.strip().split(",")]
    passed=(not compute.stdout.strip() and values[0]<=50 and values[1]<=1536 and values[2]<=80 and values[3]<=60)
    good=good+1 if passed else 0
    if good>=5:
        raise SystemExit(0)
    time.sleep(1)
raise RuntimeError("GPU idle gate timed out")
PY

set +e
"$PYTHON" "$LEDGER" \
  --label x2_privileged_teacher_reachability_v2b \
  --resource-output "$RESOURCE" --log "$LOG" --disk-path "$ROOT" \
  --timeout-seconds 300 --term-grace-seconds 5 -- \
  "$PYTHON" "$RUNNER" --prereg "$PREREG" --source "$SOURCE" \
  --report "$REPORT" --failure "$FAILURE" --num-envs 64 --seed 780042 \
  --steps 200 --device cuda:0 --headless
status=$?
set -e

(cd "$(dirname "$RESOURCE")" && sha256sum -c "$(basename "${RESOURCE}.sha256")")
(cd "$(dirname "$LOG")" && sha256sum -c "$(basename "${LOG}.sha256")")
if [[ "$status" -ne 0 ]]; then
  [[ -s "$FAILURE" ]] && (cd "$(dirname "$FAILURE")" && sha256sum -c "$(basename "${FAILURE}.sha256")")
  exit "$status"
fi

[[ -s "$REPORT" && ! -e "$FAILURE" ]]
(cd "$(dirname "$REPORT")" && sha256sum -c "$(basename "${REPORT}.sha256")")
jq -e '
  .schema=="x2_privileged_teacher_isaac_reachability_result_v1" and
  (.decision=="PASS_TEACHER_REACHABILITY_LOCAL_ONLY" or .decision=="FAIL_NO_TEACHER_REACHABILITY_STOP") and
  .evidence_boundary.optimizer_steps==0 and
  .evidence_boundary.checkpoint_writes==0
' "$REPORT" >/dev/null
jq -e '
  .exit_code==0 and .raw_returncode==0 and .autonomous_exit==true and
  .timed_out==false and .term_sent==false and .kill_sent==false and
  .forced_cleanup==false and .elapsed_s<=300 and
  .disk_used_delta_bytes<=536870912 and .gpu.memory_used_peak_mib<=5120
' "$RESOURCE" >/dev/null
[[ $(stat -c '%s' "$REPORT") -le 8388608 ]]
echo "Phase teacher v2b complete: $(jq -r .decision "$REPORT")"
