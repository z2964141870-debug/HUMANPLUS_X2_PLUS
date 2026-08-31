#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_NEUTRAL_STAND_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Stage5 only: no garment input and no recorded motion reference.
The robot must start suspended. After PD acquisition reaches
READY_FOR_GROUND, lower it until both feet clearly bear load, type 'ground',
wait for a stable static hold, then type 'policy'. Keep the gantry attached.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-stand
