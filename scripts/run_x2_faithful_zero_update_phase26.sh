#!/usr/bin/env bash
set -euo pipefail

# The Phase26 launcher has one mode only: the preregistered two-batch,
# zero-environment-step, zero-optimizer gate.  No physics/config override is
# accepted from the environment or command line.

REPO="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
CONFIG="${REPO}/configs/x2_faithful_zero_update_phase26.yaml"
EXPECTED_CONFIG_SHA256="88cd18aec067ef7c2dda4f79eed164fc7045602dcd789a8bf5f4a66c0b081533"

if [[ "$#" -ne 0 ]]; then
  echo "Phase26 accepts no positional arguments or live-training mode" >&2
  exit 44
fi

actual="$(sha256sum "${CONFIG}" | awk '{print $1}')"
if [[ "${actual}" != "${EXPECTED_CONFIG_SHA256}" ]]; then
  echo "Phase26 immutable config hash refused: ${actual}" >&2
  exit 43
fi

exec conda run -n x2-sonic-isaaclab python \
  "${REPO}/tools/retarget/run_x2_faithful_zero_update_phase26.py" \
  --config "${CONFIG}"
