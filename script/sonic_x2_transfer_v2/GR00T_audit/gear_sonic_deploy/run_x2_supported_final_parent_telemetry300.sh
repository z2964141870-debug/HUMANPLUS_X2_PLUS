#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_FINAL_PARENT_TELEMETRY300_START"
FINAL_WS="$SCRIPT_DIR/runtime_suspended/ws_ground_debug_20260831"
DEPLOY_BIN="$FINAL_WS/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"
MODEL="$SCRIPT_DIR/../../models/x2_sonic_frozen_g1core_lora_v2.onnx"
POWERED_LAUNCHER="$SCRIPT_DIR/run_x2_suspended_sonic.sh"

EXPECTED_BIN_SHA256="3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f"
EXPECTED_MODEL_SHA256="8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9"
EXPECTED_POWERED_LAUNCHER_SHA256="7b6edcb1e23f2e7507335622ae3f7c57dce0754ddeba289d22bb17582a268ff6"

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage: $0 $CONFIRM_TOKEN

Final-parent 300-second supported StandStill acceptance. This wrapper pins the
M1 policy-off-debug candidate and the frozen neutral_damped relative-entry
profile. The robot must remain attached to the gantry for the entire run.
EOF
    exit 2
fi

for required in "$FINAL_WS/install/setup.bash" "$DEPLOY_BIN" "$MODEL" "$POWERED_LAUNCHER"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required final-parent artifact: $required" >&2
        exit 1
    fi
done
if [[ ! -x "$DEPLOY_BIN" || ! -x "$POWERED_LAUNCHER" ]]; then
    echo "Final-parent deploy binary or powered launcher is not executable" >&2
    exit 1
fi

check_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA-256 mismatch: path=$path expected=$expected actual=$actual" >&2
        exit 1
    fi
}

check_sha256 "$DEPLOY_BIN" "$EXPECTED_BIN_SHA256"
check_sha256 "$MODEL" "$EXPECTED_MODEL_SHA256"
check_sha256 "$POWERED_LAUNCHER" "$EXPECTED_POWERED_LAUNCHER_SHA256"

export X2_ONBOT_WS="$FINAL_WS"
export X2_SONIC_WRITER_HZ=250
exec "$POWERED_LAUNCHER" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-damped-relative-300
