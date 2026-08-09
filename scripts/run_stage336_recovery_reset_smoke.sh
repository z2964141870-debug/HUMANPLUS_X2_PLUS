#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_01-17-25_stage_stand_backend_scratch_resume100_to300_v1/model_150.pt"
DATASET="/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"
DATASET_SHA="4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"

UPDATES="${1:-1}"
RUN_NAME="${2:-stage336_stand_recovery_balanced_smoke_v1}"
NUM_ENVS="${NUM_ENVS:-64}"
DEVICE="${DEVICE:-cuda:0}"
RECOVERY_FRACTION="${RECOVERY_FRACTION:-0.50}"

[[ "$UPDATES" =~ ^[1-5]$ ]] || { echo "updates must be 1--5" >&2; exit 2; }
for path in "$SOURCE" "$DATASET" "$TEMPLATE"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done

# Dedicated stand/recovery branch only. The source checkpoint is read-only;
# all outputs go to a new timestamped run and never overwrite BASE artifacts.
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$ROOT/tools:$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
conda run --no-capture-output -n x2-sonic-isaaclab \
  python "$ROOT/scripts/train_x2_stage221_official_ankle_match.py" \
  --num_envs "$NUM_ENVS" --max_iterations "$UPDATES" --seed 47 \
  --run_name "$RUN_NAME" --resume_checkpoint "$SOURCE" --weights_only_resume \
  --gait_template "$TEMPLATE" --gait_template_scale 0.15 \
  --actuator_domain ideal --official_ankle_pd \
  --self_collisions off --collision_profile sole12 \
  --profile stand_backend \
  --recovery_reset_dataset "$DATASET" \
  --recovery_reset_expected_sha256 "$DATASET_SHA" \
  --recovery_reset_fraction "$RECOVERY_FRACTION" \
  --recovery_reset_sampling balanced \
  --device "$DEVICE" --headless
