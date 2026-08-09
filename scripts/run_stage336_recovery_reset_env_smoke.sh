#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
DATASET="/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"
SHA="4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
OUTPUT="${1:-/tmp/stage336_recovery_reset_zero_update_smoke.json}"

OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$ROOT/tools:$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
conda run --no-capture-output -n x2-sonic-isaaclab \
  python "$ROOT/scripts/probe_x2_recovery_reset.py" \
  --dataset "$DATASET" --expected-sha256 "$SHA" --output "$OUTPUT" \
  --num-envs 16 --seed 47 --device cuda:0 --headless
