#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VIDEO_ROOT="${VIDEO_ROOT:-$REPO_ROOT/videos/official_x2/report}"
FONT="${FONT:-/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}"
CAPABILITY="$VIDEO_ROOT/x2_base_capability_official_stage250_final.mp4"
LIMITATION="$VIDEO_ROOT/x2_base_limitation_official_stage350_stiff_fixed_final.mp4"
GIF="$VIDEO_ROOT/x2_base_capability_official_stage250_final.gif"
RAW_STRAIGHT="$REPO_ROOT/videos/official_x2/stage250/stage250_straight.mp4"
RAW_RIGHT="$REPO_ROOT/videos/official_x2/stage250/stage250_turn_right.mp4"
RAW_LEFT="$REPO_ROOT/videos/official_x2/stage250/stage250_turn_left.mp4"
RAW_LIMITATION="$VIDEO_ROOT/x2_base_limitation_stage350_stiff_fixed_raw.mp4"
RESULT_ROOT="/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807"

for path in "$FONT" "$RAW_STRAIGHT" "$RAW_RIGHT" "$RAW_LEFT" "$RAW_LIMITATION"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$CAPABILITY" "$LIMITATION" "$GIF"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

# Fail closed on labels: the three capability recordings must be full-gate
# passes, while the Stage350 recording must preserve a full-gate failure.
python3 - "$RESULT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name in ("stage250_video_straight", "stage250_video_turn_right", "stage250_video_turn_left"):
    summary = json.loads((root / f"{name}.json").read_text())["summary"]
    if summary.get("domain") != "aimdk_x2_v1_official_mujoco" or summary.get("full_gate_pass") is not True:
        raise SystemExit(f"capability label contract failed: {name}")
lim = json.loads((root / "stage354_report_limitation_stage350_stiff_fixed.json").read_text())["summary"]
if lim.get("domain") != "aimdk_x2_v1_official_mujoco" or lim.get("full_gate_pass") is not False:
    raise SystemExit("limitation label contract failed")
PY

mkdir -p "$VIDEO_ROOT"

# The existing Stage250 recordings contain viewer loading before the 14.2 s
# physical episode.  Remove loading only; keep the episode at real-time speed.
ffmpeg -hide_banner -loglevel warning \
  -i "$RAW_STRAIGHT" -i "$RAW_RIGHT" -i "$RAW_LEFT" \
  -filter_complex "
    [0:v]trim=start=7.800,setpts=PTS-STARTPTS,fps=30,
      drawtext=fontfile=${FONT}:text='CAPABILITY | Official AimDK v1.0 MuJoCo | frozen Stage219 | FULL GATE PASS':x=18:y=18:fontsize=25:fontcolor=white:box=1:boxcolor=0x075985@0.80,
      drawtext=fontfile=${FONT}:text='STRAIGHT | command vx=+0.30 m/s, wz=0':x=18:y=55:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.65,
      drawtext=fontfile=${FONT}:text='RECORDED SEQUENCE  STAND - START - STRAIGHT - STOP / HOLD':x=(w-text_w)/2:y=h-62:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.70[s];
    [1:v]trim=start=6.800,setpts=PTS-STARTPTS,fps=30,
      drawtext=fontfile=${FONT}:text='CAPABILITY | Official AimDK v1.0 MuJoCo | frozen Stage219 | FULL GATE PASS':x=18:y=18:fontsize=25:fontcolor=white:box=1:boxcolor=0x075985@0.80,
      drawtext=fontfile=${FONT}:text='TURN RIGHT | command vx=+0.30 m/s, wz=+0.15 rad/s':x=18:y=55:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.65,
      drawtext=fontfile=${FONT}:text='RECORDED SEQUENCE  STAND - START - TURN RIGHT - STOP / HOLD':x=(w-text_w)/2:y=h-62:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.70[r];
    [2:v]trim=start=7.300,setpts=PTS-STARTPTS,fps=30,
      drawtext=fontfile=${FONT}:text='CAPABILITY | Official AimDK v1.0 MuJoCo | frozen Stage219 | FULL GATE PASS':x=18:y=18:fontsize=25:fontcolor=white:box=1:boxcolor=0x075985@0.80,
      drawtext=fontfile=${FONT}:text='TURN LEFT | command vx=+0.30 m/s, wz=-0.09 rad/s':x=18:y=55:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.65,
      drawtext=fontfile=${FONT}:text='RECORDED SEQUENCE  STAND - START - TURN LEFT - STOP / HOLD':x=(w-text_w)/2:y=h-62:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.70[l];
    [s][r][l]concat=n=3:v=1:a=0[out]
  " -map '[out]' -c:v libx264 -preset medium -crf 19 -pix_fmt yuv420p \
  -movflags +faststart "$CAPABILITY"

# Keep the complete physical failure and its post-fall held frames.  Only the
# initial black/loading interval is trimmed; playback remains 1x real time.
ffmpeg -hide_banner -loglevel warning -i "$RAW_LIMITATION" \
  -vf "trim=start=8.3,setpts=PTS-STARTPTS,fps=30,
    drawtext=fontfile=${FONT}:text='LIMITATION | Stage350 matched-event stiff x1.2 | frozen weights | FULL GATE FAIL':x=18:y=18:fontsize=25:fontcolor=white:box=1:boxcolor=0x9c1c1c@0.85,
    drawtext=fontfile=${FONT}:text='vx=+0.30 m/s | fixed upper | start - straight - stop':x=18:y=55:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.65,
    drawtext=fontfile=${FONT}:text='FULL RECORDED PHYSICAL SEQUENCE - FAILURE END PRESERVED':x=(w-text_w)/2:y=h-62:fontsize=27:fontcolor=white:box=1:boxcolor=0x9c1c1c@0.82" \
  -c:v libx264 -preset medium -crf 19 -pix_fmt yuv420p -movflags +faststart "$LIMITATION"

# Lightweight preview; MP4 remains the authoritative real-time artifact.
ffmpeg -hide_banner -loglevel warning -i "$CAPABILITY" \
  -vf "fps=6,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" \
  -loop 0 "$GIF"

for path in "$CAPABILITY" "$LIMITATION" "$GIF"; do
  echo "$path"
  ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,r_frame_rate \
    -of default=nw=1 "$path"
done
