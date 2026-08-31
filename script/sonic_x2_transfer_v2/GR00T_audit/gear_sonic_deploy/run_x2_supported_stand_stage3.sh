#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_STAND_STAGE3_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Supported Sonic validation after stage2 saturated the 0.10rad leg clamp:
  suspended cold start -> static ground contact -> 60-second Sonic policy
  -> gradual partial load -> automatic static return -> re-suspend

The leg target envelope increases to 0.25rad. Output LPF and target slew are
disabled so the supported probe matches the policy's trained 50Hz dynamics.
Measured-speed and tilt trips remain active.
The gantry must stay ready to catch the robot. Type stop for early return.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-stand-stage3
