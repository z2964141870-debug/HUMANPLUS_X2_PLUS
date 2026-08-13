#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/yu/x2_teleop_final/x2_sonic
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
PREREG="$ROOT/reports/retarget/x2_locomotion_zero_hold_confirmation_prereg.json"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt"
STATIONARY=/home/yu/projects/ZHY/x2_official_rl_deploy_v1/models/stand_backend_scratch_i150_actor.onnx
RESULT="$ROOT/reports/retarget/x2_locomotion_zero_hold_confirmation_result.json"
MARKDOWN="$ROOT/reports/retarget/x2_locomotion_zero_hold_confirmation.md"
LEDGER="$ROOT/tools/retarget/run_with_gpu_deadline_ledger_phase76.py"

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}"
export CWI_UPPER_MOTION="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
export CWI_UPPER_ZERO_FRACTION=1.0 CWI_UPPER_DETERMINISTIC_SPLIT=0
export CWI_UPPER_SPLIT_MODE=contiguous CWI_UPPER_SCALE=0.25
export CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1
export CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20

verify_sidecar() {
  local path="$1"
  (cd "$(dirname "$path")" && sha256sum -c "$(basename "$path").sha256")
}

verify_sidecar "$PREREG"
for path in "$RESULT" "$RESULT.sha256" "$MARKDOWN" "$MARKDOWN.sha256"; do
  test ! -e "$path"
done
free_before=$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d ' ')
test "$free_before" -ge 32212254720

isaac_campaign_before=$(du -s -B1 /tmp/IsaacLab 2>/dev/null | awk '{print $1}')
isaac_campaign_before=${isaac_campaign_before:-0}
screens=()
resources=()
seeds=(786101 786102 786103)

for seed_index in 0 1 2; do
  seed=${seeds[$seed_index]}
  prefix="$ROOT/reports/retarget/x2_locomotion_zero_hold_confirmation_seed${seed_index}"
  screen="${prefix}_screen.json"
  failure="${prefix}_failure.json"
  resource="${prefix}_resource.json"
  log="/tmp/x2_locomotion_zero_hold_confirmation_seed${seed_index}.log"
  for path in "$screen" "$screen.sha256" "$failure" "$failure.sha256" \
              "$resource" "$resource.sha256" "$log" "$log.sha256"; do
    test ! -e "$path"
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU compute process already active before seed $seed_index" >&2
    exit 1
  fi
  free_launch_before=$(df -B1 --output=avail "$ROOT" | tail -n1 | tr -d ' ')
  test "$free_launch_before" -ge 32212254720
  isaac_before=$(du -s -B1 /tmp/IsaacLab 2>/dev/null | awk '{print $1}')
  isaac_before=${isaac_before:-0}

  set +e
  "$PY" "$LEDGER" \
    --label "x2_locomotion_zero_hold_confirmation_seed${seed_index}" \
    --resource-output "$resource" --log "$log" \
    --disk-path "$ROOT" --timeout-seconds 300 --term-grace-seconds 5 -- \
    "$PY" "$ROOT/scripts/run_x2_locomotion_zero_hold_confirmation.py" \
    --prereg "$PREREG" --source "$SOURCE" --stationary "$STATIONARY" \
    --report "$screen" --failure "$failure" --seed-index "$seed_index" \
    --seed "$seed" --num-envs 256 --steps 820 --device cuda:0 --headless
  ledger_status=$?
  set -e

  verify_sidecar "$resource"
  verify_sidecar "$log"
  test "$ledger_status" -eq 0
  test -f "$screen"
  verify_sidecar "$screen"
  jq -e '.segment_status == "SEGMENT_VALID" and
         ([.technical_checks[]] | all) and
         .optimizer_steps == 0 and .backward_calls == 0 and
         .checkpoint_writes == 0 and .reward_inspected == false' "$screen" >/dev/null
  jq -e --arg label "x2_locomotion_zero_hold_confirmation_seed${seed_index}" '
         .label == $label and .exit_code == 0 and .raw_returncode == 0 and
         .autonomous_exit == true and .timed_out == false and
         .term_sent == false and .kill_sent == false and .forced_cleanup == false and
         .elapsed_s <= 300 and .gpu != null and
         .gpu.memory_used_peak_mib <= 8192 and .disk_after.free_bytes >= 32212254720' \
         "$resource" >/dev/null
  test "$(stat -c %s "$screen")" -le 67108864
  isaac_after=$(du -s -B1 /tmp/IsaacLab 2>/dev/null | awk '{print $1}')
  isaac_after=${isaac_after:-0}
  isaac_delta=$((isaac_after - isaac_before))
  if [ "$isaac_delta" -gt 268435456 ]; then
    echo "IsaacLab temp delta too large for seed $seed_index: $isaac_delta" >&2
    exit 1
  fi
  campaign_delta=$((isaac_after - isaac_campaign_before))
  if [ "$campaign_delta" -gt 805306368 ]; then
    echo "Campaign IsaacLab temp delta too large: $campaign_delta" >&2
    exit 1
  fi
  screens+=("$screen")
  resources+=("$resource")
  echo "Seed $seed_index complete; IsaacLab temp delta=$isaac_delta bytes"
done

"$PY" "$ROOT/tools/retarget/finalize_x2_locomotion_zero_hold_confirmation.py" \
  --prereg "$PREREG" \
  --screen "${screens[0]}" --screen "${screens[1]}" --screen "${screens[2]}" \
  --resource "${resources[0]}" --resource "${resources[1]}" --resource "${resources[2]}" \
  --result "$RESULT" --markdown "$MARKDOWN"
verify_sidecar "$RESULT"
verify_sidecar "$MARKDOWN"
jq -e '.optimizer_steps == 0 and .backward_calls == 0 and .checkpoint_writes == 0 and
       .permissions.training_unlocked == false and .permissions.deployment_unlocked == false' \
       "$RESULT" >/dev/null
echo "Confirmation complete: $(jq -r .decision "$RESULT")"
