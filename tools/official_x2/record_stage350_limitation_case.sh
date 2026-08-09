#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VIDEO_ROOT="${VIDEO_ROOT:-$REPO_ROOT/videos/official_x2/report}"
OUTPUT="${1:-$VIDEO_ROOT/x2_base_limitation_stage350_stiff_fixed_raw.mp4}"
RESULT_ROOT="${RESULT_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807}"
DOMAIN="${ROS_DOMAIN_ID:-94}"
CASE="stage354_report_limitation_stage350_stiff_fixed"
RESULT="$RESULT_ROOT/$CASE.json"

mkdir -p "$VIDEO_ROOT" "$RESULT_ROOT/video_logs"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT" >&2; exit 2; }
[[ ! -e "$RESULT" ]] || { echo "refusing to overwrite $RESULT" >&2; exit 2; }

# Exact Stage350 matched-event/stiff limitation contract.  This is a frozen
# Stage306 moving actor plus the frozen scratch stand actor; no update occurs.
COMMON=(
  CASE_NAME="$CASE" ROS_DOMAIN_ID="$DOMAIN" RESULT_ROOT="$RESULT_ROOT"
  MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx
  STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx
  RECOVERY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx
  COMMAND_VX=0.30 STATE_QOS_DEPTH=10 STATE_PREDICTION_SECONDS=0
  PD_PROFILE=official_kp_ankle PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2
  MOVE_SECONDS=5.2 MOVE_ACCELERATE_SECONDS=1.0 STOP_SECONDS=8.0
  STATIONARY_CONTROLLER=policy STOP_CONTROLLER=curriculum_then_policy
  STOP_TRANSITION_SECONDS=2.0 STOP_INTENT_DECELERATE_SECONDS=2.0
  FUTURE_STOP_PREVIEW_SECONDS=0.5 STOP_BRAKE_GAIN=1.5
  STOP_BRAKE_LIMIT=0.30 STOP_BRAKE_TEMPLATE_SPEED=0.30
  STOP_BRAKE_TEMPLATE_FLOOR=0.25
  HEADING_GAIN=0.0 ACTION_BIAS_MODE=lateral_recovery_supervisor
  ACTION_BIAS=0.6 ANKLE_ROLL_COMMON_BIAS=0.2
  LATERAL_POSITION_GAIN=0.8 LATERAL_VELOCITY_GAIN=0.2
  RECOVERY_ENTER_M=0.08 RECOVERY_EXIT_M=0.03 RECOVERY_SLEW_RATE_PER_S=1.0
  RECORD_X11_TCP=true MAX_ATTEMPTS=1 TIMEOUT_SECONDS=70
  REPORT_SCENE_XML="$REPO_ROOT/assets/official_x2/scene_report.xml"
)

env "${COMMON[@]}" bash "$SCRIPT_DIR/run_official_gate_case.sh" \
  >"$RESULT_ROOT/video_logs/${CASE}.log" 2>&1 &
CASE_PID=$!
cleanup() {
  kill "${FFMPEG_PID:-}" "${CASE_PID:-}" 2>/dev/null || true
  wait "${FFMPEG_PID:-}" "${CASE_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 250); do
  nc -z 127.0.0.1 6099 2>/dev/null && break
  kill -0 "$CASE_PID" 2>/dev/null || { wait "$CASE_PID"; exit $?; }
  sleep 0.1
done
nc -z 127.0.0.1 6099 2>/dev/null || { echo "X11 display did not open" >&2; exit 3; }

ffmpeg -hide_banner -loglevel warning -y \
  -f x11grab -draw_mouse 0 -framerate 30 -video_size 1280x720 \
  -i 127.0.0.1:99.0 -c:v libx264 -preset veryfast -crf 20 \
  -pix_fmt yuv420p -movflags +faststart "$OUTPUT" &
FFMPEG_PID=$!

set +e
wait "$CASE_PID"
CASE_STATUS=$?
kill -INT "$FFMPEG_PID" 2>/dev/null
wait "$FFMPEG_PID"
FFMPEG_STATUS=$?
set -e
trap - EXIT INT TERM

# The official gate wrapper exits 2 when the rollout is physically valid but
# fails its capability gate.  That is the expected outcome for this limitation
# recording, not infrastructure failure.
if [[ "$CASE_STATUS" -ne 2 ]]; then
  tail -100 "$RESULT_ROOT/video_logs/${CASE}.log" >&2
  echo "expected official limitation status 2, got $CASE_STATUS" >&2
  exit 4
fi
if [[ "$FFMPEG_STATUS" -ne 0 && "$FFMPEG_STATUS" -ne 255 ]]; then
  echo "ffmpeg exited with unexpected status $FFMPEG_STATUS" >&2
  exit "$FFMPEG_STATUS"
fi
[[ -s "$OUTPUT" ]] || { echo "recording is empty: $OUTPUT" >&2; exit 5; }

python3 - "$RESULT" <<'PY'
import json, math, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))["summary"]
expected = {
    "domain": "aimdk_x2_v1_official_mujoco",
    "model": "/models/stage306_s2652_transition_head_actor.onnx",
    "stationary_model": "/models/stand_backend_scratch_i150_actor.onnx",
    "recovery_model": "/models/stand_backend_scratch_i150_actor.onnx",
    "pd_profile": "official_kp_ankle",
    "stop_controller": "curriculum_then_policy",
    "action_bias_mode": "lateral_recovery_supervisor",
}
for key, value in expected.items():
    if s.get(key) != value:
        raise SystemExit(f"contract mismatch {key}: {s.get(key)!r} != {value!r}")
for key, value in {
    "pd_kp_multiplier": 1.2,
    "pd_kd_multiplier": 1.2,
    "command_vx_mps": 0.3,
    "move_seconds": 5.2,
    "move_accelerate_seconds": 1.0,
    "stop_seconds": 8.0,
}.items():
    if not math.isclose(float(s[key]), value, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(f"contract mismatch {key}: {s[key]!r} != {value!r}")
if s.get("full_gate_pass") is not False:
    raise SystemExit("limitation rollout unexpectedly passed; do not label this recording as failure")
PY

ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUTPUT"
echo "expected limitation gate failure preserved: $RESULT"
