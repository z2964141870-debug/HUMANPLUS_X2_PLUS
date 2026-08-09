#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
MANIFEST="$ROOT/manifests/x2_phase19_outcome_aware_state_role.json"
MANIFEST_SHA="16b8ee0539e917a9346ee5ea39ced408e871cabb5dbae19d2a8c91e2b8c485ea"
OUTDIR="$ROOT/results/phase20_outcome_aware_zero_update"
mkdir -p "$OUTDIR"

for role in success_safe critical_from_failure; do
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ROOT/tools:$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
  conda run --no-capture-output -n x2-sonic-isaaclab \
    python "$ROOT/scripts/probe_x2_phase20_outcome_aware_reset.py" \
    --manifest "$MANIFEST" --expected-manifest-sha256 "$MANIFEST_SHA" \
    --state-role "$role" --output "$OUTDIR/${role}.json" \
    --num-envs 16 --seed 47 --device cuda:0 --headless
  python3 -c 'import json,sys; payload=json.load(open(sys.argv[1])); raise SystemExit(0 if payload.get("passed") is True else 2)' \
    "$OUTDIR/${role}.json"
done
