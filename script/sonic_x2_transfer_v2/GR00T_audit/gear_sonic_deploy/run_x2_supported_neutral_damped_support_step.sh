#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIRM_TOKEN="X2_SUPPORTED_NEUTRAL_DAMPED_SUPPORT_STEP_START"
CANDIDATE_WS="$SCRIPT_DIR/runtime_suspended/ws_support_step_20260831_v3"
DEPLOY_BIN="$CANDIDATE_WS/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"
MODEL="$SCRIPT_DIR/../../models/x2_sonic_frozen_g1core_lora_v2.onnx"
POWERED_LAUNCHER="$SCRIPT_DIR/run_x2_suspended_sonic.sh"
HANDOFF_LAUNCHER="$SCRIPT_DIR/deploy_x2.sh"

EXPECTED_BIN_SHA256="9d4916a8e5fd64e4ac3ddb0f0a32903ccc90b9e86b4e146cc6cabefe63695483"
EXPECTED_MODEL_SHA256="8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9"
EXPECTED_POWERED_LAUNCHER_SHA256="247e02c28fe4fee8d5746f5b2c1bef758b10c64f8598e5d23fe803c21deee6a6"
EXPECTED_HANDOFF_LAUNCHER_SHA256="d16d52db6763edec280238bfad83b91f04b76f4fa7d66831307b5bd9ea9c63b4"

VERIFY_ONLY=false
if [[ "${1:-}" == "--verify-only" ]]; then
    VERIFY_ONLY=true
elif [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat >&2 <<EOF
Usage:
  $0 --verify-only
  $0 $CONFIRM_TOKEN

One marked support-transfer probe using the frozen neutral_damped controller.
The first 20 policy seconds keep support fixed. An accepted 'support_step'
command records one marker before one small manual hoist adjustment, followed
by a fixed 20-second observation plateau and automatic return to static PD.
This launcher never actuates the hoist and does not authorize freehand release.
EOF
    exit 2
fi

for required in \
    "$CANDIDATE_WS/install/setup.bash" \
    "$DEPLOY_BIN" \
    "$MODEL" \
    "$POWERED_LAUNCHER" \
    "$HANDOFF_LAUNCHER"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required support-step artifact: $required" >&2
        exit 1
    fi
done

if [[ ! -x "$DEPLOY_BIN" \
      || ! -x "$POWERED_LAUNCHER" \
      || ! -x "$HANDOFF_LAUNCHER" ]]; then
    echo "Support-step binary or launcher is not executable" >&2
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
check_sha256 "$HANDOFF_LAUNCHER" "$EXPECTED_HANDOFF_LAUNCHER_SHA256"

if $VERIFY_ONLY; then
    echo "SUPPORT_STEP_ARTIFACTS_VERIFIED: no process started"
    exit 0
fi

unset X2_ADOPT_PAUSED_MC X2_ADOPT_DEBUG_PORT X2_OBS_DUMP_PATH
export X2_ONBOT_WS="$CANDIDATE_WS"
export X2_SONIC_WRITER_HZ=250
exec "$POWERED_LAUNCHER" \
    X2_SUSPENDED_SONIC_START \
    --supported-neutral-damped-relative-support-step
