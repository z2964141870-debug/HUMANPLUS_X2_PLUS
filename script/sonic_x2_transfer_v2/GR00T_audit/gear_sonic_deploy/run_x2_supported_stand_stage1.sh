#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_STAND_STAGE1_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Bounded Sonic stand validation:
  suspended cold start -> static ground load -> 60-second Sonic policy
  -> automatic static return -> re-suspend -> restore MC

The gantry must remain taut. Full walking and unrestricted CONTROL remain
disabled. Type stop for an early return.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-stand-stage1
