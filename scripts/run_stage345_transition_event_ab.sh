#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
SOURCE="$ROOT/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_11-18-41_stage306_s2642_transition_head_h1p0_long15_lr1e4_v1/model_2652.pt"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"
UPPER_MOTION="$OLD/motion_lib_x2/accad_gmr_100_keypoints_v4_safe06_filtered_relaxed_split/Male1General_c3d__General_A3_-_Swing_Arms_While_Stand_stageii.pkl"

BRANCH="${1:-aligned}"
UPDATES="${UPDATES:-1}"
if ! [[ "$UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "UPDATES must be a positive integer" >&2
  exit 2
fi
case "$BRANCH" in
  control)
    PHASE_CONSISTENT=0
    RANDOM_EPISODE_PHASE=1
    PHASE_OFFSET=8.0
    STAND_S=1.0
    CRUISE_S=4.0
    ;;
  aligned)
    PHASE_CONSISTENT=1
    RANDOM_EPISODE_PHASE=0
    PHASE_OFFSET=0.0
    STAND_S=1.0
    CRUISE_S=4.0
    ;;
  handoff_control)
    PHASE_CONSISTENT=0
    RANDOM_EPISODE_PHASE=1
    PHASE_OFFSET=7.2
    STAND_S=0.0
    CRUISE_S=4.2
    ;;
  handoff_aligned)
    PHASE_CONSISTENT=1
    RANDOM_EPISODE_PHASE=0
    PHASE_OFFSET=0.0
    STAND_S=0.0
    CRUISE_S=4.2
    ;;
  *)
    echo "usage: $0 {control|aligned|handoff_control|handoff_aligned}" >&2
    exit 2
    ;;
esac

for path in "$SOURCE" "$TEMPLATE" "$UPPER_MOTION"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done

RUN_NAME="stage345_transition_event_${BRANCH}_u${UPDATES}_seed47_v1"

# Matched A/B from the frozen Stage306 checkpoint.  Both branches use one
# 10.24-second PPO rollout, exact stiff lower/waist gain, fixed upper body and
# the same terminal double-support objective.  The only principal variable is
# whether physical reset time and command/gait event time are phase-consistent.
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
CWI_UPPER_MOTION_LIST="$UPPER_MOTION" CWI_UPPER_RANDOMIZE_CLIP=0 CWI_UPPER_LOOP=1 \
CWI_UPPER_ZERO_FRACTION=1.0 \
CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_MAX_EXCURSION_RAD=0.12 \
CWI_UPPER_MAX_VELOCITY_RADPS=0.40 CWI_STAGE6_ADAPTER_MODE=future \
CWI_STAGE6_RESPONSE_ADAPTER=1 CWI_STAGE6_TRANSITION_ADAPTER=1 \
CWI_STAGE6_TRANSITION_UPPER_CONDITIONED=0 CWI_STAGE6_TRANSITION_CURRICULUM=1 \
CWI_STAGE6_PHASE_CONSISTENT_EVENT="$PHASE_CONSISTENT" \
CWI_STAGE6_RANDOM_EPISODE_PHASE="$RANDOM_EPISODE_PHASE" \
CWI_STAGE6_TRANSITION_PHASE_OFFSET_MAX_S="$PHASE_OFFSET" \
CWI_STAGE6_TRANSITION_STAND_S="$STAND_S" CWI_STAGE6_TRANSITION_ACCELERATE_S=1.0 \
CWI_STAGE6_TRANSITION_CRUISE_S="$CRUISE_S" CWI_STAGE6_TRANSITION_DECELERATE_S=2.0 \
CWI_STAGE6_TRANSITION_TERMINAL_HOLD_S=2.0 CWI_STAGE6_TERMINAL_CONTACT_REWARD=1 \
CWI_STAGE6_COORDINATION_BLEND=1.0 CWI_STAGE6_VELOCITY_MIN=0.30 \
CWI_STAGE6_VELOCITY_MAX=0.30 CWI_STAGE6_GAIN_MIN=1.20 CWI_STAGE6_GAIN_MAX=1.20 \
CWI_STAGE6_SAVE_INTERVAL=1 CWI_STAGE6_STEPS_PER_ENV=512 \
CWI_STAGE6_LEARNING_RATE="${LEARNING_RATE:-0.00005}" CWI_STAGE6_DESIRED_KL=0.003 \
conda run --no-capture-output -n x2-sonic-isaaclab \
  python "$ROOT/scripts/train_stage6_future_intent.py" \
  --num_envs "${NUM_ENVS:-32}" --max_iterations "$UPDATES" --seed 47 \
  --run_name "$RUN_NAME" --resume_checkpoint "$SOURCE" --weights_only_resume \
  --gait_template "$TEMPLATE" --gait_template_scale 0.15 \
  --actuator_domain ideal --self_collisions off --collision_profile sole12 \
  --ideal_heading_stiffness 1.0 --heading_env_fraction 1.0 \
  --yaw_command_abs_max 0.5 --heading_error_weight 0.5 \
  --profile privileged_teacher_phase_template_response_history \
  --device "${DEVICE:-cuda:0}" --headless
