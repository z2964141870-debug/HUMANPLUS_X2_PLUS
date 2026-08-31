#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_STAND_STAGE4_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Sonic standing test with the target envelope centered on the trained crouched
default pose. The command starts from the captured static hold, slews at no
more than 0.20rad/s, and permits 0.60rad leg motion for weight-bearing balance.
The gantry must remain ready to catch the robot. Type stop for early return.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-stand-stage4
