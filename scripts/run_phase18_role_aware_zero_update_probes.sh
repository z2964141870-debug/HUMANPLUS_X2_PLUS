#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
OLD="/home/humanplus/x2_teleop_final/x2_sonic"
MANIFEST="$ROOT/manifests/x2_phase17_role_aware_reset_curriculum.json"
MANIFEST_SHA="8904ebd256ee49279ac8011db04dc0044b2ad75ea86d85f53d6dce174cc2c7a7"
OUTDIR="$ROOT/results/phase18_role_aware_zero_update"
mkdir -p "$OUTDIR"

for role in success critical height_fail; do
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ROOT/tools:$ROOT/hooks:$ROOT/src:$OLD/sonic_x2_sandbox:${PYTHONPATH:-}" \
  conda run --no-capture-output -n x2-sonic-isaaclab \
    python "$ROOT/scripts/probe_x2_phase18_role_aware_reset.py" \
    --manifest "$MANIFEST" --expected-manifest-sha256 "$MANIFEST_SHA" \
    --outcome-role "$role" --output "$OUTDIR/${role}.json" \
    --num-envs 16 --seed 47 --device cuda:0 --headless
  # Isaac Kit may close cleanly after the probe records a failed gate.  Treat
  # the machine-readable result, not only the process status, as authoritative.
  python3 -c 'import json,sys; payload=json.load(open(sys.argv[1])); raise SystemExit(0 if payload.get("passed") is True else 2)' \
    "$OUTDIR/${role}.json"
done
