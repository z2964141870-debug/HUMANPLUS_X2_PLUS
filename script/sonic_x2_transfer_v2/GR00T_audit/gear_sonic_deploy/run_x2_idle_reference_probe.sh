#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_IDLE_REFERENCE_PROBE_START"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Correct-reference, suspended Sonic check:
  x2_idle_stand.x2m2 -> 1-second policy ramp -> 5-second maximum pulse
  -> 1.5-second automatic return to static PD

Target deviation remains tightly bounded:
  leg=0.02rad waist=0.02rad arm=0.03rad head=0.03rad

This is a powered test. Keep the robot firmly suspended and keep the
emergency-stop control ready. It does not test load-bearing balance.
EOF
    exit 2
fi

exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-policy-probe
