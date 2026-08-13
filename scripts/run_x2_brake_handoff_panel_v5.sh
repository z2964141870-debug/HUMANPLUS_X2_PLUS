#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
PREREG="$ROOT/reports/retarget/x2_brake_handoff_panel_v5_prereg.json"
SOURCE=/home/yu/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt
STATIONARY="$ROOT/../x2_official_rl_deploy_v1/models/stand_backend_scratch_i150_actor.onnx"
SCREEN="$ROOT/reports/retarget/x2_brake_handoff_panel_v5_screen.json"
FAILURE="$ROOT/reports/retarget/x2_brake_handoff_panel_v5_failure.json"
RESOURCE="$ROOT/reports/retarget/x2_brake_handoff_panel_v5_resource.json"
RESULT="$ROOT/reports/retarget/x2_brake_handoff_panel_v5_result.json"
MARKDOWN="$ROOT/reports/retarget/x2_brake_handoff_panel_v5.md"
LOG=/tmp/x2_brake_handoff_panel_v5.log
LEDGER="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"

verify_sidecar() {
  local path="$1"
  (cd "$(dirname "$path")" && sha256sum -c "$(basename "$path").sha256")
}

verify_sidecar "$PREREG"
for path in "$SCREEN" "$SCREEN.sha256" "$FAILURE" "$FAILURE.sha256" \
            "$RESOURCE" "$RESOURCE.sha256" "$RESULT" "$RESULT.sha256" \
            "$MARKDOWN" "$MARKDOWN.sha256" "$LOG" "$LOG.sha256"; do
  test ! -e "$path"
done
free_before=$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d ' ')
test "$free_before" -ge 30064771072
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process already active" >&2
  exit 1
fi
isaac_before=$(du -s -B1 /tmp/IsaacLab 2>/dev/null | awk '{print $1}')
isaac_before=${isaac_before:-0}

set +e
PYTHONPATH="$ROOT/src:$ROOT" "$PY" "$LEDGER" \
  --label x2_brake_handoff_panel_v5 \
  --resource-output "$RESOURCE" --log "$LOG" \
  --disk-path "$ROOT" --timeout-seconds 1200 --term-grace-seconds 5 -- \
  "$PY" "$ROOT/scripts/run_x2_brake_handoff_panel_v5.py" \
  --prereg "$PREREG" --source "$SOURCE" --stationary "$STATIONARY" \
  --report "$SCREEN" --failure "$FAILURE" --num-envs 256 --seed 785001 \
  --steps 820 --device cuda:0 --headless
ledger_status=$?
set -e

verify_sidecar "$RESOURCE"
verify_sidecar "$LOG"
test "$ledger_status" -eq 0
test -f "$SCREEN"
verify_sidecar "$SCREEN"
decision=$(jq -r '.decision' "$SCREEN")
case "$decision" in
  PASS_HANDOFF_CONTROLLER_OFFICIAL_PANEL_PREREG_ONLY|PARTIAL_HANDOFF_BRAKE_SKILL_PREREG_ONLY|FAIL_HANDOFF_NEW_ACTOR_PREREG_ONLY) ;;
  *) echo "unexpected screen decision: $decision" >&2; exit 1 ;;
esac
jq -e '.exit_code == 0 and .raw_returncode == 0 and .autonomous_exit == true and
       .timed_out == false and .term_sent == false and .kill_sent == false and
       .forced_cleanup == false and .elapsed_s <= 1200 and
       .gpu != null and .gpu.memory_used_peak_mib <= 8192 and
       .disk_after.free_bytes >= 30064771072' "$RESOURCE" >/dev/null
test "$(stat -c %s "$SCREEN")" -le 67108864
isaac_after=$(du -s -B1 /tmp/IsaacLab 2>/dev/null | awk '{print $1}')
isaac_after=${isaac_after:-0}
isaac_delta=$((isaac_after - isaac_before))
if [ "$isaac_delta" -gt 268435456 ]; then
  echo "IsaacLab-attributable temp delta too large: $isaac_delta" >&2
  exit 1
fi

PYTHONPATH="$ROOT/src:$ROOT" "$PY" "$ROOT/tools/retarget/finalize_x2_brake_handoff_panel_v5.py" \
  --prereg "$PREREG" --screen "$SCREEN" --resource "$RESOURCE" \
  --result "$RESULT" --markdown "$MARKDOWN"
verify_sidecar "$RESULT"
verify_sidecar "$MARKDOWN"
jq -e --arg decision "$decision" '.decision == $decision' "$RESULT" >/dev/null
echo "Phase v5 complete: $decision; Isaac temp delta=$isaac_delta bytes"
