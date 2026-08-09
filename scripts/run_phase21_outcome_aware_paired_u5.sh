#!/usr/bin/env bash
set -euo pipefail

# BASE Phase21 pre-registered paired smoke. The commands differ only in the
# Phase20 outcome-aware reset fraction (plus output labels). This trainer is an
# isolated stand/recovery backend: the moving Stage306 actor is never loaded.
ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
OFFICIAL_ROOT="/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1"
SOURCE="$OLD/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_01-17-25_stage_stand_backend_scratch_resume100_to300_v1/model_150.pt"
MANIFEST="$ROOT/manifests/x2_phase19_outcome_aware_state_role.json"
TEMPLATE="$OLD/data/processed/x2_official_forward_gait_phase_template_15dof.npz"
TRAINER="$ROOT/scripts/train_x2_stage221_official_ankle_match.py"

SOURCE_SHA="4da931cf8ab1f094056a9d3bd024ef555b2c94944f09fe38924d7fde10a52251"
MANIFEST_SHA="16b8ee0539e917a9346ee5ea39ced408e871cabb5dbae19d2a8c91e2b8c485ea"
TEMPLATE_SHA="16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d"
TRAINER_SHA="71aa97ae97dc57f26296c4ee37dabbfd24be7baaad7040127ccbb98a2004751d"
OUTCOME_CONTRACT_SHA="49ae05c1e70d42b2449118ea1a4588fed8f4907bd4b0aba85849105ea2cd2a0a"
STATEFUL_GLUE_SHA="ae2555dd96139ebd6b4689ab6f8ef1ecd76beca95c37afb73fc5c6198b93453e"

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
check_sha "$MANIFEST_SHA" "$MANIFEST"
check_sha "$TEMPLATE_SHA" "$TEMPLATE"
check_sha "$TRAINER_SHA" "$TRAINER"
check_sha "$OUTCOME_CONTRACT_SHA" "$ROOT/tools/official_x2/outcome_aware_state_role_v2.py"
check_sha "$STATEFUL_GLUE_SHA" "$ROOT/tools/official_x2/stateful_recovery_isaac.py"

run_one() {
  local label="$1"
  local fraction="$2"
  local run_name="phase21_outcome_aware_${label}_u5_seed47_v1"
  if find "$ROOT/logs/rsl_rl/x2_lower_velocity_flat" -maxdepth 1 -type d \
      -name "*_${run_name}" -print -quit | grep -q .; then
    echo "refusing to overwrite/resume existing Phase21 run: $run_name" >&2
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
    --outcome_aware_reset_manifest "$MANIFEST" \
    --outcome_aware_reset_manifest_sha256 "$MANIFEST_SHA" \
    --outcome_aware_reset_fraction "$fraction" \
    --device "$DEVICE" --headless
}

echo "Phase21 frozen pair: source=$SOURCE_SHA manifest=$MANIFEST_SHA seed=$SEED envs=$NUM_ENVS updates=$UPDATES"
run_one f000 0.00
run_one f005 0.05

for label in f000 f005; do
  run_dir="$(find "$ROOT/logs/rsl_rl/x2_lower_velocity_flat" -maxdepth 1 -type d \
    -name "*_phase21_outcome_aware_${label}_u5_seed47_v1" -print | sort | tail -1)"
  checkpoint="$run_dir/model_155.pt"
  [[ -f "$checkpoint" ]] || { echo "missing final checkpoint: $checkpoint" >&2; exit 2; }
  PYTHONDONTWRITEBYTECODE=1 conda run -n x2-sonic-isaaclab \
    python "$ROOT/tools/official_x2/export_rsl_actor_onnx.py" \
    --checkpoint "$checkpoint" \
    --output "$OFFICIAL_ROOT/models/phase21_outcome_aware_${label}_u5_actor.onnx"
done

echo "Phase21 paired 5-update smoke and ONNX export complete. No 25-update run is authorized."
