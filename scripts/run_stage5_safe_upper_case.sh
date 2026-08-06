#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
EVAL="$OLD/scripts/eval_x2_stage172_lower_velocity.py"
CHECKPOINT="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"
REPORT_DIR="$ROOT/reports/stage5_safe_upper"

MOTION_PATH="${1:?motion path required}"
MOTION_TAG="${2:?motion tag required}"
START_S="${3:-0.0}"
SEED="${4:-42}"
VX="${5:-0.30}"
STEPS="${6:-400}"

SCALE="0.25"
TIME_SCALE="0.5"
MAX_EXCURSION="${CWI_STAGE5_MAX_EXCURSION:-0.12}"
MAX_VELOCITY="${CWI_STAGE5_MAX_VELOCITY:-0.20}"
TILT_FALLBACK="${CWI_STAGE5_TILT_FALLBACK:-0.35}"
HEIGHT_FALLBACK="${CWI_STAGE5_HEIGHT_FALLBACK:-0.58}"
HEADING_FALLBACK="${CWI_STAGE5_HEADING_FALLBACK:-inf}"
LATCH_FALLBACK="${CWI_STAGE5_LATCH_FALLBACK:-0}"
VARIANT="${CWI_STAGE5_VARIANT:-stage5}"
DOMAIN="delay"

mkdir -p "$REPORT_DIR"
VX_TAG="$(printf '%s' "$VX" | tr '.' 'p')"
CASE_TAG="seed${SEED}_vx${VX_TAG}_${DOMAIN}_s${STEPS}"
CONTROL_STEM="stage5_control_${CASE_TAG}"
CANDIDATE_STEM="${VARIANT}_${MOTION_TAG}_${CASE_TAG}"

COMMON=(
  --checkpoint "$CHECKPOINT"
  --command_vx "$VX"
  --steps "$STEPS"
  --seed "$SEED"
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
  --actuator_filter_strength 1.0
  --actuator_delay_strength 1.0
  --headless
)

if [[ ! -f "$REPORT_DIR/${CONTROL_STEM}.json" ]]; then
  conda run --no-capture-output -n x2-sonic-isaaclab \
    env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
    PYTHONPATH="$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
    python "$EVAL" "${COMMON[@]}" \
    --stage_label "$CONTROL_STEM" \
    --output "$REPORT_DIR/${CONTROL_STEM}.json" \
    --trace_output "$REPORT_DIR/${CONTROL_STEM}.npz"
fi

conda run --no-capture-output -n x2-sonic-isaaclab \
  env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
  CWI_UPPER_MOTION="$MOTION_PATH" \
  CWI_UPPER_SCALE="$SCALE" \
  CWI_UPPER_START_S="$START_S" \
  CWI_UPPER_TIME_SCALE="$TIME_SCALE" \
  CWI_UPPER_MAX_EXCURSION_RAD="$MAX_EXCURSION" \
  CWI_UPPER_MAX_VELOCITY_RADPS="$MAX_VELOCITY" \
  CWI_UPPER_TILT_FALLBACK_RAD="$TILT_FALLBACK" \
  CWI_UPPER_HEIGHT_FALLBACK_M="$HEIGHT_FALLBACK" \
  CWI_UPPER_HEADING_FALLBACK_RAD="$HEADING_FALLBACK" \
  CWI_UPPER_LATCH_FALLBACK="$LATCH_FALLBACK" \
  python "$EVAL" "${COMMON[@]}" \
  --stage_label "$CANDIDATE_STEM" \
  --output "$REPORT_DIR/${CANDIDATE_STEM}.json" \
  --trace_output "$REPORT_DIR/${CANDIDATE_STEM}.npz"

set +e
PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" python "$ROOT/tools/analyze_stage4_upper_disturbance.py" \
  --control-json "$REPORT_DIR/${CONTROL_STEM}.json" \
  --control-trace "$REPORT_DIR/${CONTROL_STEM}.npz" \
  --candidate-json "$REPORT_DIR/${CANDIDATE_STEM}.json" \
  --candidate-trace "$REPORT_DIR/${CANDIDATE_STEM}.npz" \
  --upper-scale "$SCALE" \
  --output "$REPORT_DIR/${CANDIDATE_STEM}_comparison.json"
STAGE4_RC=$?
PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" python "$ROOT/tools/analyze_stage5_safe_upper.py" \
  --comparison "$REPORT_DIR/${CANDIDATE_STEM}_comparison.json" \
  --candidate-trace "$REPORT_DIR/${CANDIDATE_STEM}.npz" \
  --max-excursion-rad "$MAX_EXCURSION" \
  --max-velocity-radps "$MAX_VELOCITY" \
  --tilt-fallback-rad "$TILT_FALLBACK" \
  --height-fallback-m "$HEIGHT_FALLBACK" \
  --heading-fallback-rad "$HEADING_FALLBACK" \
  --output "$REPORT_DIR/${CANDIDATE_STEM}_stage5.json"
STAGE5_RC=$?
set -e

if [[ "$STAGE4_RC" -ne 0 || "$STAGE5_RC" -ne 0 ]]; then
  exit 2
fi
