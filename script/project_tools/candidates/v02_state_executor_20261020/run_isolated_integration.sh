#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
set -u
W=/agibot/data/home/agi/x2-v02-state-executor-20261021.rMYM0G
export PYTHONPATH=/agibot/data/home/agi/aimdk_msgs_pc2/local/lib/python3.10/dist-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/agibot/data/home/agi/aimdk_msgs_pc2/lib:/agibot/data/home/agi/gear-sonic/onnxruntime/lib:${LD_LIBRARY_PATH:-}
export ROS_DOMAIN_ID=173 ROS_LOCALHOST_ONLY=1
unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
mode="${1:-serial}"
extra=()
case "$mode" in
  serial) ;;
  independent) extra+=(--independent-writer) ;;
  refusal) exec /usr/bin/python3 "$W/test_offline_writer_refusal.py" --workspace "$W" ;;
  *) echo "Invalid test mode" >&2; exit 2 ;;
esac
cases=(healthy malformed stdin stale late_malformed fake_publish fake_stale fake_hold)
if [[ "$mode" == independent ]]; then cases+=(fake_block fake_partial0 fake_partial1 fake_partial2 fake_partial3); fi
result=0
for test_case in "${cases[@]}"; do
  if /usr/bin/python3 "$W/test_full_process.py" --workspace "$W" --case "$test_case" "${extra[@]}" > "$W/full-$mode-$test_case.log" 2>&1; then
    printf '%s PASS
' "$test_case"
  else
    printf '%s FAIL (see full-%s.log)
' "$test_case" "$test_case"
    result=1
  fi
done
exit "$result"
