#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OFFICIAL=/home/yu/projects/ZHY/x2_official_rl_deploy_v1
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
PREREG="$ROOT/reports/retarget/x2_official_state_feedback_brake_panel_repair_prereg_v1.json"
ATTEMPT0_RAW="$OFFICIAL/results/state_feedback_brake_official_panel_20260813"
REPAIR_RAW="$OFFICIAL/results/state_feedback_brake_official_panel_left_repair_20260813"
REPORT="$ROOT/reports/retarget"
LEDGER="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"
FINALIZER="$ROOT/tools/retarget/finalize_x2_official_state_feedback_brake_panel_repair.py"
RESULT="$REPORT/x2_official_state_feedback_brake_panel_repair_result.json"
MARKDOWN="$REPORT/x2_official_state_feedback_brake_panel_repair.md"

verify_sidecar() { (cd "$(dirname "$1")" && sha256sum -c "$(basename "$1").sha256"); }
verify_sidecar "$PREREG"
"$PY" - "$PREREG" <<'PY'
import hashlib,json,pathlib,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
for section in ('immutable_code','immutable_inputs','immutable_evidence'):
    for name,record in p[section].items():
        path=pathlib.Path(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError(f'immutable drift: {section}.{name}')
PY
test ! -e "$REPAIR_RAW"
for path in "$RESULT" "$RESULT.sha256" "$MARKDOWN" "$MARKDOWN.sha256"; do test ! -e "$path"; done
test "$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d ' ')" -ge 32212254720
if docker ps --format '{{.Names}}' | grep -q '^x2-'; then exit 1; fi
mkdir -p "$REPAIR_RAW"

repair_raws=()
repair_resources=()
repair_index=0
for speed in medium low; do
  if [ "$speed" = medium ]; then vx=0.30; else vx=0.25; fi
  for repeat in 1 2 3; do
    case_name="x2_sfbrake_repair_${speed}_turn_left_r${repeat}"
    raw="$REPAIR_RAW/$case_name.json"
    resource="$REPORT/${case_name}_resource.json"
    log="/tmp/${case_name}.log"
    for path in "$raw" "$raw.sha256" "$resource" "$resource.sha256" "$log" "$log.sha256"; do test ! -e "$path"; done
    domain=$((150 + repair_index))
    port=$((31950 + repair_index))
    set +e
    "$PY" "$LEDGER" --label "x2_official_state_feedback_repair_case${repair_index}" \
      --resource-output "$resource" --log "$log" --disk-path "$ROOT" \
      --timeout-seconds 180 --term-grace-seconds 5 -- \
      env CASE_NAME="$case_name" ROS_DOMAIN_ID="$domain" OFFICIAL_ROOT="$OFFICIAL" RESULT_ROOT="$REPAIR_RAW" \
      TEMPLATE_PATH=/home/yu/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz \
      MODEL_PATH=/models/stage219_s2600_actor.onnx COMMAND_VX="$vx" STATE_QOS_DEPTH=1 STATE_PREDICTION_SECONDS=0 \
      SIMULATOR_HTTP_PORT="$port" MOVE_SECONDS=4.0 STOP_SECONDS=8.0 STOP_CONTROLLER=brake_then_policy \
      STOP_BRAKE_GAIN=1.5 STOP_BRAKE_LIMIT=0.30 STOP_BRAKE_TEMPLATE_SPEED=0.30 STOP_BRAKE_TEMPLATE_FLOOR=0.25 \
      EVENT_HOLD_MIN_SECONDS=0.5 EVENT_HOLD_SPEED=0.05 MAX_ATTEMPTS=1 TIMEOUT_SECONDS=120 \
      FIXED_WZ=-0.09 MIRROR_POLICY=true ACTION_BIAS_MODE=turn_progress_feedback ACTION_BIAS=1.0 \
      YAW_ACTION_GAIN=2.0 ANKLE_ROLL_COMMON_BIAS=-0.20 bash "$ROOT/tools/official_x2/run_official_gate_case.sh"
    status=$?
    set -e
    case "$status" in 0|2) ;; *) exit "$status";; esac
    verify_sidecar "$resource"; verify_sidecar "$log"
    test -s "$raw"
    (cd "$REPAIR_RAW" && sha256sum "$(basename "$raw")" > "$(basename "$raw").sha256")
    verify_sidecar "$raw"
    jq -e '.summary.mirror_policy==true and .summary.stop_controller=="brake_then_policy"' "$raw" >/dev/null
    repair_raws+=("$raw"); repair_resources+=("$resource")
    repair_index=$((repair_index+1))
  done
done
test "$repair_index" -eq 6

raws=(); resources=(); repair_index=0
for speed in medium low; do
  for repeat in 1 2 3 4 5 6; do
    raws+=("$ATTEMPT0_RAW/x2_sfbrake_${speed}_straight_r${repeat}.json")
    resources+=("$REPORT/x2_sfbrake_${speed}_straight_r${repeat}_resource.json")
  done
  for repeat in 1 2 3; do
    raws+=("$ATTEMPT0_RAW/x2_sfbrake_${speed}_turn_right_r${repeat}.json")
    resources+=("$REPORT/x2_sfbrake_${speed}_turn_right_r${repeat}_resource.json")
  done
  for repeat in 1 2 3; do
    raws+=("${repair_raws[$repair_index]}"); resources+=("${repair_resources[$repair_index]}")
    repair_index=$((repair_index+1))
  done
done
args=(--prereg "$PREREG" --result "$RESULT" --markdown "$MARKDOWN")
for path in "${raws[@]}"; do args+=(--case "$path"); done
for path in "${resources[@]}"; do args+=(--resource "$path"); done
PYTHONPATH="$ROOT:$ROOT/src" "$PY" "$FINALIZER" "${args[@]}"
verify_sidecar "$RESULT"; verify_sidecar "$MARKDOWN"
jq -e '.decision=="COMPLETE_REPAIRED_PANEL_DIAGNOSTIC_NO_PROMOTION" and .runs==24 and .repair_runs==6 and .optimizer_steps==0 and .permissions.training_unlocked==false and .permissions.deployment_unlocked==false' "$RESULT" >/dev/null
