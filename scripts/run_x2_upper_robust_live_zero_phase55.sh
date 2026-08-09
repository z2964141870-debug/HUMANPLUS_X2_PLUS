#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
OLD=/home/humanplus/x2_teleop_final/x2_sonic
UPPER="$ROOT/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
OUT="$ROOT/reports/retarget/x2_upper_robust_lower_live_zero_phase55.json"

OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
CWI_UPPER_MOTION="$UPPER" CWI_UPPER_ZERO_FRACTION=0.50 \
CWI_UPPER_DETERMINISTIC_SPLIT=1 \
CWI_UPPER_SPLIT_MODE=interleaved \
CWI_UPPER_SCALE=0.25 CWI_UPPER_TIME_SCALE=1.0 CWI_UPPER_LOOP=1 \
CWI_UPPER_MAX_EXCURSION_RAD=0.12 CWI_UPPER_MAX_VELOCITY_RADPS=0.20 \
conda run --no-capture-output -n x2-sonic-isaaclab \
  python "$ROOT/scripts/run_x2_upper_robust_live_zero_phase55.py" \
  --num_envs 64 --seed 42 --device cuda:0 --headless --output "$OUT"
