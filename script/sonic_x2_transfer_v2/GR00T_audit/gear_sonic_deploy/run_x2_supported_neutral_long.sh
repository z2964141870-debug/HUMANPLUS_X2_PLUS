#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_NEUTRAL_LONG_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Five-minute neutral Sonic balance test. No garment input and no recorded
motion reference. Start suspended, keep the gantry attached, and keep a person
ready to support the robot. After a return to static PD, 'policy' may be used
again. Type 'stop' at any sign of oscillation or noise; type 'lifted' only to
end the session and restore MC after the robot is suspended.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-long
