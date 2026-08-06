#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
REPORT_DIR="$ROOT/reports/stage5_safe_upper"
LOG_DIR="$REPORT_DIR/logs"
STEPS="${1:-400}"
VARIANT="stage5c_heading025"

mkdir -p "$LOG_DIR"

MOTIONS=(
  "/home/humanplus/x2_teleop_final/x2_sonic/motion_lib_x2/receiver_safe_upper_body_fixed_feet_split/wave__wave_20260708_211740.pkl|wave_real|5.0"
  "/home/humanplus/x2_teleop_final/x2_sonic/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Male1General_c3d__General_A3_-_Swing_Arms_While_Stand_stageii.pkl|swing_arms|0.0"
  "/home/humanplus/x2_teleop_final/x2_sonic/motion_lib_x2/bmlrub_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/rub073__0013_knocking1_stageii.pkl|knocking|0.0"
  "/home/humanplus/x2_teleop_final/x2_sonic/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Male2General_c3d__A6-_Box_lift_stageii.pkl|box_lift|0.0"
)

for motion_spec in "${MOTIONS[@]}"; do
  IFS="|" read -r motion_path motion_tag start_s <<<"$motion_spec"
  for vx in "0.30" "0.20"; do
    vx_tag="${vx/./p}"
    stem="${VARIANT}_${motion_tag}_seed42_vx${vx_tag}_delay_s${STEPS}"
    result="$REPORT_DIR/${stem}_stage5.json"
    log="$LOG_DIR/${stem}.log"
    if [[ -f "$result" ]]; then
      printf 'CACHED %s\n' "$stem"
      continue
    fi
    printf 'RUN %s\n' "$stem"
    set +e
    CWI_STAGE5_VARIANT="$VARIANT" \
    CWI_STAGE5_HEADING_FALLBACK="0.25" \
    CWI_STAGE5_LATCH_FALLBACK="1" \
      bash "$ROOT/scripts/run_stage5_safe_upper_case.sh" \
      "$motion_path" "$motion_tag" "$start_s" 42 "$vx" "$STEPS" \
      >"$log" 2>&1
    rc=$?
    set -e
    printf 'DONE %s rc=%s\n' "$stem" "$rc"
  done
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" \
  python "$ROOT/tools/summarize_stage5_safe_upper.py" \
  --report-dir "$REPORT_DIR" \
  --steps "$STEPS" \
  --expected 8 \
  --variant-prefix "$VARIANT" \
  --output-json "$REPORT_DIR/${VARIANT}_panel_s${STEPS}.json" \
  --output-md "$REPORT_DIR/${VARIANT}_panel_s${STEPS}.md"
