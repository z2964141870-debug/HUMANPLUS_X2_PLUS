#!/usr/bin/env bash
set -euo pipefail

# This script does not start a controller. It only requests the guarded proxy
# to leave anchored StandStill after every fixed-parent and runtime gate passes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_ARM_SUPPORTED_GARMENT_LIVE_REFERENCE"
MANIFEST="${X2_FIXED_PARENT_MANIFEST:-$SCRIPT_DIR/approved_manifests/x2_fixed_standstill_parent.json}"
STATUS_FILE="${X2_LIVE_GATE_STATUS:-/tmp/x2_live_reference_gate.status.json}"
PYTHON_BIN="${X2_GATE_PYTHON:-/agibot/data/home/agi/miniconda3/envs/teleop/bin/python}"
REQUESTER="$SCRIPT_DIR/scripts/garment_zmq/request_live_reference_arm.py"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Requests a stationary-only, 10-second StandStill-to-live reference blend.
It does not start MC/HAL control and cannot bypass these hard gates:
  - approved fixed-StandStill parent manifest and artifact hashes;
  - three supported 30-second passes plus one supported 300-second pass;
  - live proxy state STANDSTILL_READY;
  - fresh powered x2_debug and raw reference streams.
EOF
    exit 2
fi

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python runtime: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$REQUESTER" ]] || { echo "Missing arm requester: $REQUESTER" >&2; exit 2; }

exec "$PYTHON_BIN" -u "$REQUESTER" \
    --manifest "$MANIFEST" \
    --root "$SCRIPT_DIR" \
    --status "$STATUS_FILE"
