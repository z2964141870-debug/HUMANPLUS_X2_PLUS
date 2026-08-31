#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_POLICY_PROBE_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Sequence:
  suspended -> stop MC -> cached-pose PD acquire -> default-pose hold
  -> type load with gantry taut -> type policy for a bounded 5-second probe
  -> automatic/static return -> re-suspend -> type lifted -> restore MC

Full walking and unrestricted CONTROL are disabled. Type stop to return early.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-policy-probe
