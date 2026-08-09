#!/usr/bin/env bash
set -euo pipefail

# Pre-registered BASE Phase9 paired smoke.  The two commands differ only in
# recovery_reset_fraction (and the output label required to keep artifacts
# separate).  Moving Stage306 is never loaded by this trainer.
ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
OFFICIAL_ROOT="/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_01-17-25_stage_stand_backend_scratch_resume100_to300_v1/model_150.pt"
DATASET="$OFFICIAL_ROOT/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"
SOURCE_REPORT="$ROOT/reports/official_x2/stage335_stop_recovery_state_extraction.json"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"
TRAINER="$ROOT/scripts/train_x2_stage221_official_ankle_match.py"

SOURCE_SHA="4da931cf8ab1f094056a9d3bd024ef555b2c94944f09fe38924d7fde10a52251"
DATASET_SHA="4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
SOURCE_REPORT_SHA="b4755acd98ecfa72717de7057d5ab9b23c54585dd2acd40e322dde55afc8f926"
TEMPLATE_SHA="16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d"
TRAINER_SHA="8dbb7f5f67bdac9e47247c62db8700e71aaec3612f9521e1fa5dee085d4db6e8"
STATEFUL_GLUE_SHA="36c1931bc193bd1d702ce06ef67d2dbf46b1bfe1daada93d4854ba744eb5b79f"
RESET_CONTRACT_SHA="58fc61112d4e7dba62cbea8a29c575beb66c78cc49e29a5c9d7ae3ad7a61fbaf"

NUM_ENVS="${NUM_ENVS:-64}"
DEVICE="${DEVICE:-cuda:0}"
SEED=47
UPDATES=5

check_sha() {
  local expected="$1"
  local path="$2"
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA mismatch: $path actual=$actual expected=$expected" >&2
    exit 2
  }
}

check_sha "$SOURCE_SHA" "$SOURCE"
check_sha "$DATASET_SHA" "$DATASET"
check_sha "$SOURCE_REPORT_SHA" "$SOURCE_REPORT"
check_sha "$TEMPLATE_SHA" "$TEMPLATE"
check_sha "$TRAINER_SHA" "$TRAINER"
check_sha "$STATEFUL_GLUE_SHA" "$ROOT/tools/official_x2/stateful_recovery_isaac.py"
check_sha "$RESET_CONTRACT_SHA" "$ROOT/tools/official_x2/recovery_reset_curriculum.py"

run_one() {
  local label="$1"
  local fraction="$2"
  local run_name="phase9_stateful_recovery_${label}_u5_seed47_v1"
  if find "$ROOT/logs/rsl_rl/x2_lower_velocity_flat" -maxdepth 1 -type d \
      -name "*_${run_name}" -print -quit | grep -q .; then
    echo "refusing to overwrite/resume existing Phase9 run: $run_name" >&2
    exit 2
  fi
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ROOT/tools:$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
  conda run --no-capture-output -n x2-sonic-isaaclab \
    python "$TRAINER" \
    --num_envs "$NUM_ENVS" --max_iterations "$UPDATES" --seed "$SEED" \
    --run_name "$run_name" \
    --resume_checkpoint "$SOURCE" --weights_only_resume \
    --gait_template "$TEMPLATE" --gait_template_scale 0.15 \
    --actuator_domain ideal --official_ankle_pd \
    --self_collisions off --collision_profile sole12 \
    --profile stand_backend \
    --recovery_reset_dataset "$DATASET" \
    --recovery_reset_expected_sha256 "$DATASET_SHA" \
    --recovery_reset_stateful_source_report "$SOURCE_REPORT" \
    --recovery_reset_stateful_source_report_sha256 "$SOURCE_REPORT_SHA" \
    --recovery_reset_fraction "$fraction" \
    --recovery_reset_sampling balanced \
    --device "$DEVICE" --headless
}

echo "Phase9 frozen pair: source=$SOURCE_SHA dataset=$DATASET_SHA seed=$SEED envs=$NUM_ENVS updates=$UPDATES"
run_one f000 0.00
run_one f005 0.05

for label in f000 f005; do
  run_dir="$(find "$ROOT/logs/rsl_rl/x2_lower_velocity_flat" -maxdepth 1 -type d \
    -name "*_phase9_stateful_recovery_${label}_u5_seed47_v1" -print | sort | tail -1)"
  checkpoint="$run_dir/model_155.pt"
  [[ -f "$checkpoint" ]] || { echo "missing final checkpoint: $checkpoint" >&2; exit 2; }
  PYTHONDONTWRITEBYTECODE=1 conda run -n x2-sonic-isaaclab \
    python "$ROOT/tools/official_x2/export_rsl_actor_onnx.py" \
    --checkpoint "$checkpoint" \
    --output "$OFFICIAL_ROOT/models/phase9_stateful_recovery_${label}_u5_actor.onnx"
done

echo "Phase9 training pair and ONNX export complete; official evaluation is a separate frozen command."
