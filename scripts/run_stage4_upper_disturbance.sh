#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
EVAL="$OLD/scripts/eval_x2_stage172_lower_velocity.py"
CHECKPOINT="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"
MOTION="$OLD/motion_lib_x2/receiver_safe_upper_body_fixed_feet_split/wave__wave_20260708_211740.pkl"
REPORT_DIR="$ROOT/reports/stage4_upper_disturbance"
SCALE="${1:-0.0}"
DOMAIN="${2:-delay}"
STEPS="${3:-150}"
START_S="${4:-5.0}"
ZERO_DELAY_GROUPS="${5:-}"
TIME_SCALE="${6:-1.0}"
mkdir -p "$REPORT_DIR"

COMMON=(
  --checkpoint "$CHECKPOINT"
  --command_vx 0.30
  --steps "$STEPS"
  --seed 42
  --observation_profile teacher_phase_template
  --action_scale_multiplier 2.0
  --gait_template "$TEMPLATE"
  --gait_template_scale 0.15
  --heading_hold
  --heading_stiffness 0.05
  --heading_rate_limit 0.2
  --collision_profile sole12
  --self_collisions off
  --actuator_domain "$DOMAIN"
  --actuator_response_strength 1.0
  --headless
)
if [[ "$DOMAIN" != "ideal" ]]; then
  COMMON+=(--actuator_filter_strength 1.0)
fi
if [[ "$DOMAIN" != "ideal" && "$DOMAIN" != "filter" ]]; then
  COMMON+=(--actuator_delay_strength 1.0)
fi
DOMAIN_TAG="$DOMAIN"
if [[ -n "$ZERO_DELAY_GROUPS" ]]; then
  COMMON+=(--zero_delay_groups "$ZERO_DELAY_GROUPS")
  ZERO_DELAY_TAG="$(printf '%s' "$ZERO_DELAY_GROUPS" | tr ',' '-')"
  DOMAIN_TAG="${DOMAIN}_zd${ZERO_DELAY_TAG}"
fi

CONTROL_STEM="stage4_control_${DOMAIN_TAG}_s${STEPS}"
if [[ ! -f "$REPORT_DIR/${CONTROL_STEM}.json" ]]; then
  conda run --no-capture-output -n x2-sonic-isaaclab \
    env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
    PYTHONPATH="$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
    python "$EVAL" "${COMMON[@]}" \
    --stage_label "$CONTROL_STEM" \
    --output "$REPORT_DIR/${CONTROL_STEM}.json" \
    --trace_output "$REPORT_DIR/${CONTROL_STEM}.npz" || exit $?
fi

SCALE_TAG="$(printf '%s' "$SCALE" | tr '.' 'p')"
TIME_TAG=""
if [[ "$TIME_SCALE" != "1" && "$TIME_SCALE" != "1.0" ]]; then
  TIME_SCALE_TAG="$(printf '%s' "$TIME_SCALE" | tr '.' 'p')"
  TIME_TAG="_ts${TIME_SCALE_TAG}"
fi
CANDIDATE_STEM="stage4_upper_scale${SCALE_TAG}${TIME_TAG}_${DOMAIN_TAG}_s${STEPS}"
conda run --no-capture-output -n x2-sonic-isaaclab \
  env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
  CWI_UPPER_MOTION="$MOTION" \
  CWI_UPPER_SCALE="$SCALE" \
  CWI_UPPER_START_S="$START_S" \
  CWI_UPPER_TIME_SCALE="$TIME_SCALE" \
  python "$EVAL" "${COMMON[@]}" \
  --stage_label "$CANDIDATE_STEM" \
  --output "$REPORT_DIR/${CANDIDATE_STEM}.json" \
  --trace_output "$REPORT_DIR/${CANDIDATE_STEM}.npz" || exit $?

PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" python "$ROOT/tools/analyze_stage4_upper_disturbance.py" \
  --control-json "$REPORT_DIR/${CONTROL_STEM}.json" \
  --control-trace "$REPORT_DIR/${CONTROL_STEM}.npz" \
  --candidate-json "$REPORT_DIR/${CANDIDATE_STEM}.json" \
  --candidate-trace "$REPORT_DIR/${CANDIDATE_STEM}.npz" \
  --upper-scale "$SCALE" \
  --output "$REPORT_DIR/${CANDIDATE_STEM}_comparison.json"
