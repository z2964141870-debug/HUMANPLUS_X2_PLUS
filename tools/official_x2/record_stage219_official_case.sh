#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SKILL="${1:?usage: $0 straight|turn_right|turn_left [output.mp4]}"
VIDEO_ROOT="${VIDEO_ROOT:-$REPO_ROOT/videos/official_x2}"
OUTPUT="${2:-$VIDEO_ROOT/stage219_v5_${SKILL}.mp4}"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
DOMAIN="${ROS_DOMAIN_ID:-90}"
CASE="stage219_v5_video_${SKILL}"

mkdir -p "$VIDEO_ROOT" "$RESULT_ROOT/video_logs"
rm -f "$OUTPUT"

COMMON=(
  CASE_NAME="$CASE" ROS_DOMAIN_ID="$DOMAIN" RESULT_ROOT="$RESULT_ROOT"
  MODEL_PATH=/models/stage219_s2600_actor.onnx COMMAND_VX=0.30
  MOVE_SECONDS=4.0 STOP_SECONDS=8.0 STOP_CONTROLLER=policy
  RECORD_X11_TCP=true MAX_ATTEMPTS=1 TIMEOUT_SECONDS=70
  REPORT_SCENE_XML="$REPO_ROOT/assets/official_x2/scene_report.xml"
)
case "$SKILL" in
  straight)
    EXTRA=(ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.50
      ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M=0.12 RECOVERY_EXIT_M=0.04
      RECOVERY_SLEW_RATE_PER_S=1.0)
    ;;
  turn_right)
    EXTRA=(FIXED_WZ=0.15 ACTION_BIAS_MODE=hip_yaw_common ACTION_BIAS=0.50
      ANKLE_ROLL_COMMON_BIAS=0.20)
    ;;
  turn_left)
    EXTRA=(MIRROR_POLICY=true FIXED_WZ=-0.09 ACTION_BIAS_MODE=hip_yaw_common
      ACTION_BIAS=-0.50 ANKLE_ROLL_COMMON_BIAS=-0.20)
    ;;
  *) echo "unknown skill: $SKILL" >&2; exit 2 ;;
esac

env "${COMMON[@]}" "${EXTRA[@]}" bash "$SCRIPT_DIR/run_official_gate_case.sh" \
  >"$RESULT_ROOT/video_logs/${CASE}.log" 2>&1 &
CASE_PID=$!
cleanup() {
  kill "${FFMPEG_PID:-}" "${CASE_PID:-}" 2>/dev/null || true
  wait "${FFMPEG_PID:-}" "${CASE_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 200); do
  nc -z 127.0.0.1 6099 2>/dev/null && break
  kill -0 "$CASE_PID" 2>/dev/null || { wait "$CASE_PID"; exit $?; }
  sleep 0.1
done
nc -z 127.0.0.1 6099 2>/dev/null || { echo "X11 display did not open" >&2; exit 3; }

ffmpeg -hide_banner -loglevel warning -y \
  -f x11grab -draw_mouse 0 -framerate 30 -video_size 1280x720 \
  -i 127.0.0.1:99.0 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$OUTPUT" &
FFMPEG_PID=$!

set +e
wait "$CASE_PID"
CASE_STATUS=$?
kill -INT "$FFMPEG_PID" 2>/dev/null
wait "$FFMPEG_PID"
FFMPEG_STATUS=$?
set -e
trap - EXIT INT TERM

[[ "$CASE_STATUS" -eq 0 ]] || { tail -80 "$RESULT_ROOT/video_logs/${CASE}.log" >&2; exit "$CASE_STATUS"; }
[[ -s "$OUTPUT" ]] || { echo "recording is empty: $OUTPUT" >&2; exit 4; }
ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUTPUT"
exit 0
