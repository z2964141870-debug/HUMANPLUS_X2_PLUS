#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"

MODE="${1:?adapter mode required: current, future, or future_no_phase}"
ITERATIONS="${2:?number of PPO updates required}"
RUN_NAME="${3:?run name required}"
DEVICE="${4:-cpu}"
RESUME_CHECKPOINT="${5:-$SOURCE}"
COORDINATION_BLEND="${6:-1.0}"
HEADING_ERROR_WEIGHT="${7:-0.0}"
SAVE_INTERVAL="${8:-1}"

case "$MODE" in
  current|future|future_no_phase) ;;
  *)
    echo "invalid Stage6 mode: $MODE" >&2
    exit 2
    ;;
esac
if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "iterations must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "resume checkpoint not found: $RESUME_CHECKPOINT" >&2
  exit 2
fi

MOTIONS=(
  "$OLD/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Male1General_c3d__General_A3_-_Swing_Arms_While_Stand_stageii.pkl"
  "$OLD/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Female1General_c3d__A3_-_Swing_t2_stageii.pkl"
  "$OLD/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Male2General_c3d__A6-_Box_lift_stageii.pkl"
  "$OLD/motion_lib_x2/bmlrub_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/rub073__0013_knocking1_stageii.pkl"
  "$OLD/motion_lib_x2/kit_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/572__wave_both09_stageii.pkl"
  "$OLD/motion_lib_x2/kit_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/674__shower_right_arm03_stageii.pkl"
)
MOTION_LIST="$(IFS=';'; printf '%s' "${MOTIONS[*]}")"

OMNI_KIT_ACCEPT_EULA=YES \
ACCEPT_EULA=Y \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
CWI_UPPER_MOTION_LIST="$MOTION_LIST" \
CWI_UPPER_RANDOMIZE_CLIP=1 \
CWI_UPPER_LOOP=1 \
CWI_UPPER_SCALE=0.25 \
CWI_UPPER_TIME_SCALE=0.5 \
CWI_UPPER_MAX_EXCURSION_RAD=0.12 \
CWI_UPPER_MAX_VELOCITY_RADPS=0.20 \
CWI_STAGE6_ADAPTER_MODE="$MODE" \
CWI_STAGE6_COORDINATION_BLEND="$COORDINATION_BLEND" \
CWI_STAGE6_VELOCITY_MIN=0.20 \
CWI_STAGE6_VELOCITY_MAX=0.45 \
CWI_STAGE6_SAVE_INTERVAL="$SAVE_INTERVAL" \
conda run --no-capture-output -n x2-sonic-isaaclab \
  python "$ROOT/scripts/train_stage6_future_intent.py" \
  --num_envs 32 \
  --max_iterations "$ITERATIONS" \
  --seed 42 \
  --run_name "$RUN_NAME" \
  --resume_checkpoint "$RESUME_CHECKPOINT" \
  --weights_only_resume \
  --gait_template "$TEMPLATE" \
  --gait_template_scale 0.15 \
  --actuator_domain delay \
  --actuator_response_strength 1.0 \
  --actuator_filter_strength 1.0 \
  --actuator_delay_strength 1.0 \
  --actuator_ideal_fraction 0.75 \
  --self_collisions off \
  --collision_profile sole12 \
  --ideal_heading_stiffness 1.0 \
  --heading_env_fraction 1.0 \
  --yaw_command_abs_max 0.5 \
  --heading_error_weight "$HEADING_ERROR_WEIGHT" \
  --response_heading_stiffness 0.05 \
  --profile privileged_teacher_phase_template_response_history \
  --device "$DEVICE" \
  --headless
