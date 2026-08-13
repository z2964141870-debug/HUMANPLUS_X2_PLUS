#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
PYTHON=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
RUNNER="$ROOT/scripts/run_x2_privileged_teacher_panel_v2d.py"
LEDGER="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_privileged_teacher_panel_v2d.py"
PREREG="$ROOT/reports/retarget/x2_privileged_teacher_panel_v2d_prereg.json"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
FINAL="$ROOT/reports/retarget/x2_privileged_teacher_panel_v2d_result.json"
FINAL_MD="$ROOT/reports/retarget/x2_privileged_teacher_panel_v2d.md"

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25
export CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20

(cd "$(dirname "$PREREG")" && sha256sum -c "$(basename "${PREREG}.sha256")")
for path in "$FINAL" "${FINAL}.sha256" "$FINAL_MD"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

gpu_idle() {
  "$PYTHON" - <<'PY'
import subprocess, time
deadline=time.monotonic()+180
good=0
while time.monotonic()<deadline:
    compute=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
    row=subprocess.run(["nvidia-smi","--id=0","--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
    values=[float(x.strip()) for x in row.stdout.strip().split(",")]
    passed=(not compute.stdout.strip() and values[0]<=50 and values[1]<=1536 and values[2]<=80 and values[3]<=60)
    good=good+1 if passed else 0
    if good>=5: raise SystemExit(0)
    time.sleep(1)
raise RuntimeError("GPU idle gate timed out")
PY
}

cumulative=0
for seed in 781001 781002 781003; do
  for lane in A B; do
    launch="s${seed}_${lane}"
    report="$ROOT/reports/retarget/x2_privileged_teacher_panel_v2d_${launch}_report.json"
    failure="$ROOT/reports/retarget/x2_privileged_teacher_panel_v2d_${launch}_failure.json"
    resource="$ROOT/reports/retarget/x2_privileged_teacher_panel_v2d_${launch}_resource.json"
    log="/tmp/x2_privileged_teacher_panel_v2d_${launch}.log"
    for path in "$report" "${report}.sha256" "$failure" "${failure}.sha256" \
      "$resource" "${resource}.sha256" "$log" "${log}.sha256"; do
      [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
    done
    free_bytes=$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')
    [[ "$free_bytes" -ge 34359738368 ]]
    gpu_idle
    set +e
    "$PYTHON" "$LEDGER" \
      --label "x2_privileged_teacher_panel_v2d_${launch}" \
      --resource-output "$resource" --log "$log" --disk-path "$ROOT" \
      --timeout-seconds 180 --term-grace-seconds 5 -- \
      "$PYTHON" "$RUNNER" --prereg "$PREREG" --source "$SOURCE" \
      --report "$report" --failure "$failure" --num-envs 128 --seed "$seed" \
      --lane "$lane" --steps 512 --device cuda:0 --headless
    status=$?
    set -e
    (cd "$(dirname "$resource")" && sha256sum -c "$(basename "${resource}.sha256")")
    (cd "$(dirname "$log")" && sha256sum -c "$(basename "${log}.sha256")")
    if [[ "$status" -ne 0 ]]; then
      [[ -s "$failure" ]] && (cd "$(dirname "$failure")" && sha256sum -c "$(basename "${failure}.sha256")")
      exit "$status"
    fi
    [[ -s "$report" && ! -e "$failure" ]]
    (cd "$(dirname "$report")" && sha256sum -c "$(basename "${report}.sha256")")
    jq -e '.schema=="x2_privileged_teacher_panel_launch_v1" and .decision=="PANEL_LAUNCH_FINITE" and ([.technical_checks[]]|all)' "$report" >/dev/null
    jq -e '.exit_code==0 and .raw_returncode==0 and .autonomous_exit==true and .timed_out==false and .term_sent==false and .kill_sent==false and .forced_cleanup==false and .elapsed_s<=180 and .disk_used_delta_bytes<=536870912 and .gpu.memory_used_peak_mib<=8192' "$resource" >/dev/null
    delta=$(jq -r '.disk_used_delta_bytes | if .>0 then . else 0 end' "$resource")
    cumulative=$((cumulative + delta))
    [[ "$cumulative" -le 2147483648 ]]
    echo "completed $launch cumulative_disk_delta=$cumulative"
  done
done

"$PYTHON" "$FINALIZER" --prereg "$PREREG" --result "$FINAL" --markdown "$FINAL_MD"
(cd "$(dirname "$FINAL")" && sha256sum -c "$(basename "${FINAL}.sha256")")
echo "Panel v2d complete: $(jq -r .decision "$FINAL")"
