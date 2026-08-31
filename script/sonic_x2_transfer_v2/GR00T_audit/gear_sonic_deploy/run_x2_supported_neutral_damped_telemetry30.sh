#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_NEUTRAL_DAMPED_TELEMETRY30_START"
TELEMETRY_WS="$SCRIPT_DIR/runtime_suspended/ws_telemetry_20260830"
EXPECTED_BIN_SHA256="7fa51a4df2553fcd52158dfb62fe6bb7dfa71e79ea51b718bd4f086e94efe40c"
DEPLOY_BIN="$TELEMETRY_WS/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Telemetry-only 30-second reproduction of fixed StandStill neutral_damped.
Frozen controller: 0.12rad/s supported target slew and 250Hz HAL writer.
EOF
    exit 2
fi

if [[ ! -f "$TELEMETRY_WS/install/setup.bash" ]]; then
    echo "Missing isolated telemetry workspace: $TELEMETRY_WS/install/setup.bash" >&2
    exit 1
fi
if [[ ! -x "$DEPLOY_BIN" ]]; then
    echo "Missing telemetry deploy binary: $DEPLOY_BIN" >&2
    exit 1
fi
ACTUAL_BIN_SHA256="$(sha256sum "$DEPLOY_BIN" | awk '{print $1}')"
if [[ "$ACTUAL_BIN_SHA256" != "$EXPECTED_BIN_SHA256" ]]; then
    echo "Telemetry deploy binary SHA-256 mismatch: expected=$EXPECTED_BIN_SHA256 actual=$ACTUAL_BIN_SHA256" >&2
    exit 1
fi

export X2_ONBOT_WS="$TELEMETRY_WS"
export X2_SONIC_WRITER_HZ=250
exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-damped-30
