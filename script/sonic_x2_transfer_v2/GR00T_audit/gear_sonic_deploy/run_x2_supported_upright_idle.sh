#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_UPRIGHT_IDLE_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Repeatable supported Sonic test using the recorded idle-stand joint sequence
with root roll/pitch corrected to upright. Custom static PD remains active
between policy attempts. Type 'lifted' only after re-suspending to restore MC.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-upright-idle
