#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_RELATIVE_NEUTRAL_DAMPED_ANKLEP40_ENTRY135_30_START"
TELEMETRY_WS="$SCRIPT_DIR/runtime_suspended/ws_relative_entry_20260830"
EXPECTED_BIN_SHA256="e9bfe4d5fcb330802a9f1ccbef85a0fc27370605653bac04a12aeb2159886aec"
DEPLOY_BIN="$TELEMETRY_WS/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Thirty-second fixed StandStill, relative-entry probe. Ankle-pitch Kp remains
40.0 and every control parameter remains frozen. This wrapper only tightens
the measured ankle-pitch entry-error gate from 0.150 to 0.135 rad, excluding
the repeated 0.144 rad toe-rise entry while retaining all recorded passes.
Keep the robot protected by the gantry throughout the run.
EOF
    exit 2
fi

if [[ ! -f "$TELEMETRY_WS/install/setup.bash" ]]; then
    echo "Missing isolated relative-entry workspace: $TELEMETRY_WS/install/setup.bash" >&2
    exit 1
fi
if [[ ! -x "$DEPLOY_BIN" ]]; then
    echo "Missing relative-entry deploy binary: $DEPLOY_BIN" >&2
    exit 1
fi
ACTUAL_BIN_SHA256="$(sha256sum "$DEPLOY_BIN" | awk '{print $1}')"
if [[ "$ACTUAL_BIN_SHA256" != "$EXPECTED_BIN_SHA256" ]]; then
    echo "Relative-entry deploy binary SHA-256 mismatch: expected=$EXPECTED_BIN_SHA256 actual=$ACTUAL_BIN_SHA256" >&2
    exit 1
fi

export X2_ONBOT_WS="$TELEMETRY_WS"
export X2_SONIC_WRITER_HZ=250
exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-damped-anklep40-entry135-relative-30
