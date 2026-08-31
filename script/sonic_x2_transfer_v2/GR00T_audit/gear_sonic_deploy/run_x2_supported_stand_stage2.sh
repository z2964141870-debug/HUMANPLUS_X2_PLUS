#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_STAND_STAGE2_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Partial-load Sonic validation after stage1:
  suspended cold start -> static ground contact -> 60-second Sonic policy
  -> gradual partial load -> automatic static return -> re-suspend

Global 0.10rad target and 0.5rad/s measured-speed safety limits remain active.
The gantry must stay ready to catch the robot. Type stop for early return.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-stand-stage2
