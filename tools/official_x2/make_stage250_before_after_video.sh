#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OLD_ROOT="${OLD_ROOT:-$REPO_ROOT/videos/official_x2}"
NEW_ROOT="${NEW_ROOT:-$REPO_ROOT/videos/official_x2/stage250}"
OUTPUT="${1:-$NEW_ROOT/stage219_v5_vs_stage250_official_mujoco_three_skill.mp4}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf}"

[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT" >&2; exit 2; }
for path in \
  "$OLD_ROOT/stage219_v5_straight.mp4" "$NEW_ROOT/stage250_straight.mp4" \
  "$OLD_ROOT/stage219_v5_turn_right.mp4" "$NEW_ROOT/stage250_turn_right.mp4" \
  "$OLD_ROOT/stage219_v5_turn_left.mp4" "$NEW_ROOT/stage250_turn_left.mp4"; do
  [[ -s "$path" ]] || { echo "missing input video: $path" >&2; exit 2; }
done

mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -hide_banner -loglevel warning \
  -i "$OLD_ROOT/stage219_v5_straight.mp4" -i "$NEW_ROOT/stage250_straight.mp4" \
  -i "$OLD_ROOT/stage219_v5_turn_right.mp4" -i "$NEW_ROOT/stage250_turn_right.mp4" \
  -i "$OLD_ROOT/stage219_v5_turn_left.mp4" -i "$NEW_ROOT/stage250_turn_left.mp4" \
  -filter_complex "
    [0:v]fps=30,scale=640:360,setpts=PTS-STARTPTS,drawtext=fontfile=${FONT}:text='BEFORE - wrong action feedback':x=18:y=18:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[o0];
    [1:v]fps=30,scale=640:360,setpts=PTS-STARTPTS,drawtext=fontfile=${FONT}:text='AFTER - RSL clip contract':x=18:y=18:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[n0];
    [o0][n0]hstack=inputs=2:shortest=1,drawtext=fontfile=${FONT}:text='STRAIGHT':x=(w-text_w)/2:y=h-38:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.55[s];
    [2:v]fps=30,scale=640:360,setpts=PTS-STARTPTS,drawtext=fontfile=${FONT}:text='BEFORE - wrong action feedback':x=18:y=18:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[o1];
    [3:v]fps=30,scale=640:360,setpts=PTS-STARTPTS,drawtext=fontfile=${FONT}:text='AFTER - RSL clip contract':x=18:y=18:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[n1];
    [o1][n1]hstack=inputs=2:shortest=1,drawtext=fontfile=${FONT}:text='TURN RIGHT':x=(w-text_w)/2:y=h-38:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.55[r];
    [4:v]fps=30,scale=640:360,setpts=PTS-STARTPTS,drawtext=fontfile=${FONT}:text='BEFORE - wrong action feedback':x=18:y=18:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[o2];
    [5:v]fps=30,scale=640:360,setpts=PTS-STARTPTS,drawtext=fontfile=${FONT}:text='AFTER - RSL clip contract':x=18:y=18:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65[n2];
    [o2][n2]hstack=inputs=2:shortest=1,drawtext=fontfile=${FONT}:text='TURN LEFT':x=(w-text_w)/2:y=h-38:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.55[l];
    [s][r][l]concat=n=3:v=1:a=0[out]
  " \
  -map '[out]' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUTPUT"

ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUTPUT"
