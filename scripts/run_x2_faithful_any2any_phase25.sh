#!/usr/bin/env bash
set -euo pipefail

# Phase25 is the only guarded entrypoint for future faithful live work.
# It accepts only a positional mode and no asset/physics environment override.
# Phase24 still refuses live-zero pending explicit post-Phase25 review.

REPO="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
MANIFEST="${REPO}/configs/x2_faithful_physics_phase25.yaml"
RUNTIME="${REPO}/configs/x2_faithful_physics_phase25_runtime.json"
EXPECTED_MANIFEST_SHA256="cc02be069637a488a7330928ab8500794f7972531ea093dc2be8cd839e247dbb"
MODE="${1:-audit}"

conda run -n x2-sonic-isaaclab python \
  "${REPO}/tools/retarget/guard_x2_physics_phase25.py" \
  --manifest "${MANIFEST}" \
  --runtime "${RUNTIME}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}"

exec "${REPO}/scripts/run_x2_faithful_any2any_phase24.sh" "${MODE}"
