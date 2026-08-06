#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
BASE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"

CHECKPOINT="${1:-$BASE}"
MODE="${2:-disabled}"
LABEL="${3:-stage6_base_panel_s400}"
STEPS="${4:-400}"
DEVICE="${5:-cpu}"
BLEND="${6:-1.0}"
REPORT_SUBDIR="${7:-stage6_future_intent}"
SEED="${8:-42}"
INITIAL_ROLL_PITCH_RANGE="${9:-0.0}"
INITIAL_YAW_RANGE="${10:-0.0}"
INITIAL_LATERAL_VELOCITY_RANGE="${11:-0.0}"
INITIAL_YAW_RATE_RANGE="${12:-0.0}"
RECORD_DIAGNOSTIC_TRACES="${13:-0}"
if [[ ! "$REPORT_SUBDIR" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "report subdirectory must be one safe path component" >&2
  exit 2
fi
OUTPUT="$ROOT/reports/$REPORT_SUBDIR/${LABEL}.json"

case "$MODE" in
  disabled|current|future|future_no_phase) ;;
  *)
    echo "invalid Stage6 adapter mode: $MODE" >&2
    exit 2
    ;;
esac
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi
if [[ "$RECORD_DIAGNOSTIC_TRACES" != "0" && "$RECORD_DIAGNOSTIC_TRACES" != "1" ]]; then
  echo "record diagnostic traces must be 0 or 1" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

MOTIONS=(
  "$OLD/motion_lib_x2/receiver_safe_upper_body_fixed_feet_split/wave__wave_20260708_211740.pkl"
  "$OLD/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Male1General_c3d__General_A3_-_Swing_Arms_While_Stand_stageii.pkl"
  "$OLD/motion_lib_x2/bmlrub_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/rub073__0013_knocking1_stageii.pkl"
  "$OLD/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Male2General_c3d__A6-_Box_lift_stageii.pkl"
  "$OLD/motion_lib_x2/kit_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/572__wave_left01_stageii.pkl"
  "$OLD/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Female1General_c3d__A6-_lift_box_t2_stageii.pkl"
)
MOTION_LIST="$(IFS=';'; printf '%s' "${MOTIONS[*]}")"
TRACE_ARGS=()
if [[ "$RECORD_DIAGNOSTIC_TRACES" == "1" ]]; then
  TRACE_ARGS+=(--record-diagnostic-traces)
fi

OMNI_KIT_ACCEPT_EULA=YES \
ACCEPT_EULA=Y \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
CWI_UPPER_MOTION_LIST="$MOTION_LIST" \
CWI_UPPER_START_S_LIST="5;0;0;0;0;0" \
CWI_STAGE6_PANEL_TAGS="wave_real;swing_arms;knocking;box_lift;wave_left_heldout;female_lift_heldout" \
CWI_UPPER_SCALE=0.25 \
CWI_UPPER_TIME_SCALE=0.5 \
CWI_UPPER_MAX_EXCURSION_RAD=0.12 \
CWI_UPPER_MAX_VELOCITY_RADPS=0.20 \
conda run --no-capture-output -n x2-sonic-isaaclab \
  python "$ROOT/scripts/eval_stage6_panel_batch.py" \
  --checkpoint "$CHECKPOINT" \
  --adapter-mode "$MODE" \
  --output "$OUTPUT" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --speeds 0.20 0.30 \
  --gait-template "$TEMPLATE" \
  --stage-label "$LABEL" \
  --coordination-blend "$BLEND" \
  --initial-roll-pitch-range-rad "$INITIAL_ROLL_PITCH_RANGE" \
  --initial-yaw-range-rad "$INITIAL_YAW_RANGE" \
  --initial-lateral-velocity-range-mps "$INITIAL_LATERAL_VELOCITY_RANGE" \
  --initial-yaw-rate-range-radps "$INITIAL_YAW_RATE_RANGE" \
  "${TRACE_ARGS[@]}" \
  --device "$DEVICE" \
  --headless
