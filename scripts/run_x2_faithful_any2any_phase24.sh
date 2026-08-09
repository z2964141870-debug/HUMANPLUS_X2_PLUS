#!/usr/bin/env bash
set -euo pipefail

# Immutable Phase24 entrypoint.  No environment-variable overrides are read.
# train-static/eval-static only validate their split-locked Phase23 hook.
# live-zero must fail with exit 42 until the checked-in B5 gate is frozen.

REPO="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
CONFIG="${REPO}/configs/x2_faithful_any2any_phase24.yaml"
ENTRY="${REPO}/tools/retarget/audit_x2_faithful_entrypoint_phase24.py"
MODE="${1:-audit}"

case "${MODE}" in
  audit|train-static|eval-static|live-zero)
    ;;
  *)
    echo "usage: $0 {audit|train-static|eval-static|live-zero}" >&2
    exit 64
    ;;
esac

exec conda run -n x2-sonic-isaaclab python "${ENTRY}" \
  --config "${CONFIG}" \
  --mode "${MODE}"
