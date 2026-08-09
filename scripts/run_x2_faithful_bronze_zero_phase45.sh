#!/usr/bin/env bash
set -euo pipefail

REPO="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
CONFIG="${REPO}/configs/x2_faithful_bronze_exact_s7_phase45.yaml"
EXPECTED_CONFIG_SHA256="d966a5fa9a7dab32f68178efda8551b87bda94d60803fb28d256ac056d9df296"

if [[ "$#" -ne 0 ]]; then
  echo "Phase45 zero gate accepts no optimizer/update mode" >&2
  exit 45
fi

actual="$(sha256sum "${CONFIG}" | awk '{print $1}')"
if [[ "${actual}" != "${EXPECTED_CONFIG_SHA256}" ]]; then
  echo "Phase45 immutable config hash refused: ${actual}" >&2
  exit 43
fi

exec conda run -n x2-sonic-isaaclab python \
  "${REPO}/tools/retarget/run_x2_faithful_bronze_zero_phase45.py" \
  --config "${CONFIG}"
