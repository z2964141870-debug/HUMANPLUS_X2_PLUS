#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OFFICIAL=/home/yu/projects/ZHY/x2_official_rl_deploy_v1
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
PREREG="$ROOT/reports/retarget/x2_official_state_feedback_brake_prereg_v1.json"
RAW_ROOT="$OFFICIAL/results/state_feedback_brake_then_zero_feasibility_20260813"
LEDGER="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"
RESULT="$ROOT/reports/retarget/x2_official_state_feedback_brake_result.json"
MARKDOWN="$ROOT/reports/retarget/x2_official_state_feedback_brake.md"

verify_sidecar() {
  local path="$1"
  (cd "$(dirname "$path")" && sha256sum -c "$(basename "$path").sha256")
}

verify_sidecar "$PREREG"
"$PY" - "$PREREG" <<'PY'
import hashlib, json, pathlib, sys
prereg = json.load(open(sys.argv[1], encoding="utf-8"))
for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
    for name, record in prereg[section].items():
        path = pathlib.Path(record["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise RuntimeError(f"immutable drift: {section}.{name}: {actual}")
PY
image_identity=$(sg docker -c "docker image inspect x2-aimdk-humble:1.0 --format '{{index .RepoDigests 0}}'")
test "$image_identity" = "$(jq -r .boundary.docker_image_identity "$PREREG")"
test ! -e "$RAW_ROOT"
for path in "$RESULT" "$RESULT.sha256" "$MARKDOWN" "$MARKDOWN.sha256"; do
  test ! -e "$path"
done
test "$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d ' ')" -ge 32212254720
if docker ps --format '{{.Names}}' | grep -q '^x2-'; then
  echo "another X2 official container is active" >&2
  exit 1
fi
mkdir -p "$RAW_ROOT"

cases=(
  x2_state_feedback_brake_low_straight_r4
  x2_state_feedback_brake_low_turn_right_r3
)
domains=(210 211)
ports=(31910 31911)
motions=(straight turn_right)
resources=()
raws=()

for index in 0 1; do
  case_name=${cases[$index]}
  motion=${motions[$index]}
  resource="$ROOT/reports/retarget/${case_name}_resource.json"
  log="/tmp/${case_name}.log"
  raw="$RAW_ROOT/${case_name}.json"
  for path in "$resource" "$resource.sha256" "$log" "$log.sha256" "$raw" "$raw.sha256"; do
    test ! -e "$path"
  done
  common=(
    CASE_NAME="$case_name" ROS_DOMAIN_ID="${domains[$index]}"
    OFFICIAL_ROOT="$OFFICIAL" RESULT_ROOT="$RAW_ROOT"
    TEMPLATE_PATH=/home/yu/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz
    MODEL_PATH=/models/stage219_s2600_actor.onnx COMMAND_VX=0.25
    STATE_QOS_DEPTH=1 STATE_PREDICTION_SECONDS=0 SIMULATOR_HTTP_PORT="${ports[$index]}"
    MOVE_SECONDS=4.0 STOP_SECONDS=8.0 STOP_CONTROLLER=brake_then_policy
    STOP_BRAKE_GAIN=1.5 STOP_BRAKE_LIMIT=0.30
    STOP_BRAKE_TEMPLATE_SPEED=0.30 STOP_BRAKE_TEMPLATE_FLOOR=0.25
    EVENT_HOLD_MIN_SECONDS=0.5 EVENT_HOLD_SPEED=0.05
    MAX_ATTEMPTS=1 TIMEOUT_SECONDS=120
  )
  if [ "$motion" = straight ]; then
    extra=(
      HEADING_GAIN=0.50 HEADING_RATE_LIMIT=0.10
      ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.50
      ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M=0.12
      RECOVERY_EXIT_M=0.04 RECOVERY_SLEW_RATE_PER_S=1.0
    )
  else
    extra=(
      FIXED_WZ=0.15 ACTION_BIAS_MODE=turn_progress_feedback ACTION_BIAS=1.0
      YAW_ACTION_GAIN=2.0 ANKLE_ROLL_COMMON_BIAS=0.20
    )
  fi
  set +e
  "$PY" "$LEDGER" --label "x2_official_state_feedback_brake_case${index}" \
    --resource-output "$resource" --log "$log" --disk-path "$ROOT" \
    --timeout-seconds 180 --term-grace-seconds 5 -- \
    env "${common[@]}" "${extra[@]}" bash "$ROOT/tools/official_x2/run_official_gate_case.sh"
  status=$?
  set -e
  case "$status" in 0|2) ;; *) echo "infrastructure failure for $case_name: $status" >&2; exit "$status";; esac
  verify_sidecar "$resource"
  verify_sidecar "$log"
  test -s "$raw"
  (cd "$RAW_ROOT" && sha256sum "$(basename "$raw")" > "$(basename "$raw").sha256")
  verify_sidecar "$raw"
  jq -e --arg expected_label "x2_official_state_feedback_brake_case${index}" '
    .label == $expected_label and (.raw_returncode == 0 or .raw_returncode == 2) and
    .exit_code == .raw_returncode and .timed_out == false and .term_sent == false and
    .kill_sent == false and .forced_cleanup == false and .elapsed_s <= 180 and
    .disk_after.free_bytes >= 32212254720' "$resource" >/dev/null
  jq -e '.summary.stop_controller == "brake_then_policy" and
         .summary.stop_brake_gain == 1.5 and
         .summary.stop_brake_limit_mps == 0.3 and
         .summary.stop_brake_template_speed_mps == 0.3 and
         .summary.stop_brake_template_floor == 0.25 and
         (.summary.startup_gate_pass|type)=="boolean" and
         (.summary.move_gate_pass|type)=="boolean" and
         (.summary.stop_gate_pass|type)=="boolean" and
         (.summary.full_gate_pass|type)=="boolean"' "$raw" >/dev/null
  resources+=("$resource")
  raws+=("$raw")
done

PYTHONPATH="$ROOT:$ROOT/src" "$PY" \
  "$ROOT/tools/retarget/finalize_x2_official_state_feedback_brake_failure_cases.py" \
  --prereg "$PREREG" --case "${raws[0]}" --case "${raws[1]}" \
  --resource "${resources[0]}" --resource "${resources[1]}" \
  --result "$RESULT" --markdown "$MARKDOWN"
verify_sidecar "$RESULT"
verify_sidecar "$MARKDOWN"
jq -e '.optimizer_steps==0 and .backward_calls==0 and .checkpoint_writes==0 and
       .permissions.training_unlocked==false and .permissions.deployment_unlocked==false' "$RESULT" >/dev/null
echo "Official state-feedback brake feasibility complete: $(jq -r .decision "$RESULT")"
