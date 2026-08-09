#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
SOURCE="$ROOT/logs/rsl_rl/x2_lower_velocity_flat/2026-08-09_11-14-44_stage337_recovery_f010_u5_seed47_v1/model_155.pt"
DATASET="/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"
DATASET_SHA="4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
TEMPLATE="/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz"

for path in "$SOURCE" "$DATASET" "$TEMPLATE"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done

# Single-variable continuation of the Stage337 f010 branch.  Keep the same
# curriculum, seed and plant, restore the optimizer, and add exactly 5 PPO
# updates.  This branch never overwrites the frozen source or BASE assets.
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$ROOT/tools:$ROOT/hooks:$ROOT/src:/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox:${PYTHONPATH:-}" \
conda run --no-capture-output -n x2-sonic-isaaclab \
  python "$ROOT/scripts/train_x2_stage221_official_ankle_match.py" \
  --num_envs "${NUM_ENVS:-64}" --max_iterations 5 --seed 47 \
  --run_name stage339_recovery_f010_continue5_seed47_v1 \
  --resume_checkpoint "$SOURCE" \
  --gait_template "$TEMPLATE" --gait_template_scale 0.15 \
  --actuator_domain ideal --official_ankle_pd \
  --self_collisions off --collision_profile sole12 \
  --profile stand_backend \
  --recovery_reset_dataset "$DATASET" \
  --recovery_reset_expected_sha256 "$DATASET_SHA" \
  --recovery_reset_fraction 0.10 \
  --recovery_reset_sampling balanced \
  --device "${DEVICE:-cuda:0}" --headless
