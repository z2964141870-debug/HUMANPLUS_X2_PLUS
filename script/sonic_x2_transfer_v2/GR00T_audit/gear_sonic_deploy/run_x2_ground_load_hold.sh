#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_GROUND_LOAD_HOLD_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Sequence:
  suspended -> stop MC -> cached-pose PD acquire -> default-pose hold
  -> type load for monitored, supported ground loading
  -> re-suspend -> type lifted -> restore MC

Sonic policy is hard-disabled. The command 'go' is always rejected.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --ground-load-test-only
