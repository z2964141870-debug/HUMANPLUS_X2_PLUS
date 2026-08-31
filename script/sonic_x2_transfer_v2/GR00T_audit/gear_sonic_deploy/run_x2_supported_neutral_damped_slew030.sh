#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_NEUTRAL_DAMPED_SLEW030_START"
CORRECTED_WS="$SCRIPT_DIR/runtime_suspended/ws_entrygate_8ee01b_20260829"
EXPECTED_BIN_SHA256="8ee01bc8f78d843895e5718627f36fdfd50771df1824ba06f288a0ed05b0a6a0"
DEPLOY_BIN="$CORRECTED_WS/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Corrected neutral_damped standing candidate. The sole tuning delta from the
corrected 0.12 launcher is supported target slew 0.12 -> 0.30rad/s.
This launcher is staged for a separately authorized suspended powered test.
EOF
    exit 2
fi

if [[ ! -f "$CORRECTED_WS/install/setup.bash" ]]; then
    echo "Missing isolated corrected workspace: $CORRECTED_WS/install/setup.bash" >&2
    exit 1
fi
if [[ ! -x "$DEPLOY_BIN" ]]; then
    echo "Missing corrected deploy binary: $DEPLOY_BIN" >&2
    exit 1
fi
ACTUAL_BIN_SHA256="$(sha256sum "$DEPLOY_BIN" | awk '{print $1}')"
if [[ "$ACTUAL_BIN_SHA256" != "$EXPECTED_BIN_SHA256" ]]; then
    echo "Corrected deploy binary SHA-256 mismatch: expected=$EXPECTED_BIN_SHA256 actual=$ACTUAL_BIN_SHA256" >&2
    exit 1
fi

export X2_ONBOT_WS="$CORRECTED_WS"
exec "$SCRIPT_DIR/run_x2_suspended_sonic.sh" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-damped-slew030
