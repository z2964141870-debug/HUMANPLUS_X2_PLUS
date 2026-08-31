#!/usr/bin/env bash
set -Eeuo pipefail

# Live V2 garment -> 3588S reference offload -> HMCP -> corrected C++ Sonic.
# This launcher is intentionally dry-run only: the C++ node is constructed
# without HAL command publishers and official MC remains in control.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCAP_ROOT="${X2_MOCAP_ROOT:-/agibot/data/home/agi/mocap_teleop/x2_pc1_ble_teleop}"
SMARTWEAR_ROOT="${X2_SMARTWEAR_ROOT:-/agibot/data/home/agi/projects/smartwear_v2}"
GMR_ROOT="${X2_GMR_ROOT:-$SMARTWEAR_ROOT/runtime_g1_gmr_fast_20260829}"
PYTHON_BIN="${X2_GARMENT_PYTHON:-/agibot/data/home/agi/miniconda3/envs/teleop/bin/python}"
CUDA_OVERLAY="${X2_CUDA_OVERLAY:-/agibot/data/home/agi/x2_v2_teleop/cuda126_overlay}"
CUPTI_LIB="${X2_CUPTI_LIB:-/agibot/data/home/agi/fgp_cuda_probe_FV6ZFU4O/cuda_components/cuda_cupti-linux-aarch64-12.6.80-archive/lib}"
CUDA_LIB="${X2_CUDA_LIB:-/usr/local/cuda-12.6/targets/aarch64-linux/lib}"
COMPAT_ROOT="${X2_HMCP_COMPAT_ROOT:-$SMARTWEAR_ROOT/runtime_hmcp_compat}"

GARMENT_SENDER="${X2_GARMENT_SENDER:-$SMARTWEAR_ROOT/garment_udp_v2_reference_offload.py}"
ADAPTER="$SCRIPT_DIR/scripts/garment_zmq/publish_v51_reference.py"
SUMMARY="$SCRIPT_DIR/scripts/garment_zmq/summarize_live_dryrun.py"
OFFLOAD_ROOT="$SCRIPT_DIR/scripts/offload"
DEPLOY_BIN="$SCRIPT_DIR/runtime_suspended/ws/install_gatecheck_corrected_20260829/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"
ORT_LIB="$SCRIPT_DIR/runtime_suspended/onnxruntime/lib"
MODEL="${X2_SONIC_MODEL:-$SCRIPT_DIR/../../models/x2_sonic_frozen_g1core_lora_v2.onnx}"

BLE_UP="${X2_BLE_UP:-F7:C6:1F:AB:37:E0}"
BLE_DOWN="${X2_BLE_DOWN:-DD:65:A4:4A:22:36}"
DRYRUN_SECONDS="${X2_DRYRUN_SECONDS:-30}"
SOURCE_READY_FRAMES="${X2_SOURCE_READY_FRAMES:-80}"
SOURCE_READY_TIMEOUT="${X2_SOURCE_READY_TIMEOUT:-120}"
BLE_READY_TIMEOUT="${X2_BLE_READY_TIMEOUT:-180}"
SOURCE_TARGET_HZ="${X2_SOURCE_TARGET_HZ:-30}"
SMPL_FAST_DEVICE="${X2_SMPL_FAST_DEVICE:-cpu}"
GMR_BACKEND="${X2_GMR_BACKEND:-native}"
GMR_MAX_ITER="${X2_GMR_MAX_ITER:-2}"
TORCH_THREADS="${X2_TORCH_THREADS:-1}"
REFERENCE_HOST="${X2_REFERENCE_HOST:-10.0.1.42}"
REFERENCE_PORT="${X2_REFERENCE_PORT:-51237}"
REFERENCE_TIMEOUT_MS="${X2_REFERENCE_TIMEOUT_MS:-80}"
REFERENCE_STARTUP_TIMEOUT_MS="${X2_REFERENCE_STARTUP_TIMEOUT_MS:-1000}"
REFERENCE_STARTUP_FRAMES="${X2_REFERENCE_STARTUP_FRAMES:-50}"
REFERENCE_RESET_TIMEOUT_S="${X2_REFERENCE_RESET_TIMEOUT_S:-10}"
REFERENCE_EPOCH="${X2_REFERENCE_EPOCH:-$(date +%s)}"
REFERENCE_PIPELINE="${X2_REFERENCE_PIPELINE:-1}"
EXPECT_REFERENCE_DISCONNECT="${X2_EXPECT_REFERENCE_DISCONNECT:-0}"

case "$SMPL_FAST_DEVICE" in
    cpu|cuda|auto) ;;
    *) echo "[live-dryrun] invalid X2_SMPL_FAST_DEVICE=$SMPL_FAST_DEVICE" >&2; exit 2 ;;
esac
case "$GMR_BACKEND" in
    baseline|fixed|native) ;;
    *) echo "[live-dryrun] invalid X2_GMR_BACKEND=$GMR_BACKEND" >&2; exit 2 ;;
esac
[[ "$GMR_MAX_ITER" =~ ^[0-9]+$ ]] || {
    echo "[live-dryrun] invalid X2_GMR_MAX_ITER=$GMR_MAX_ITER" >&2
    exit 2
}
[[ "$TORCH_THREADS" =~ ^[1-9][0-9]*$ ]] || {
    echo "[live-dryrun] invalid X2_TORCH_THREADS=$TORCH_THREADS" >&2
    exit 2
}
[[ "$REFERENCE_STARTUP_FRAMES" =~ ^[0-9]+$ ]] || {
    echo "[live-dryrun] invalid X2_REFERENCE_STARTUP_FRAMES=$REFERENCE_STARTUP_FRAMES" >&2
    exit 2
}
case "$REFERENCE_PIPELINE" in
    0|1) ;;
    *) echo "[live-dryrun] invalid X2_REFERENCE_PIPELINE=$REFERENCE_PIPELINE (expected 0 or 1)" >&2; exit 2 ;;
esac
case "$EXPECT_REFERENCE_DISCONNECT" in
    0|1) ;;
    *) echo "[live-dryrun] invalid X2_EXPECT_REFERENCE_DISCONNECT=$EXPECT_REFERENCE_DISCONNECT (expected 0 or 1)" >&2; exit 2 ;;
esac
reference_pipeline_args=()
if [[ "$REFERENCE_PIPELINE" == "1" ]]; then
    reference_pipeline_args+=(--reference-pipeline)
fi

for path in \
    "$PYTHON_BIN" \
    "$GARMENT_SENDER" \
    "$ADAPTER" \
    "$SUMMARY" \
    "$OFFLOAD_ROOT/reference_offload_client.py" \
    "$OFFLOAD_ROOT/reference_offload_protocol.py" \
    "$DEPLOY_BIN" \
    "$ORT_LIB/libonnxruntime.so.1" \
    "$MODEL" \
    "$SMARTWEAR_ROOT/onnx_models/LFP_dense_taichi_ft.onnx" \
    "$SMARTWEAR_ROOT/onnx_models/TIC4Clothes_dense.onnx" \
    "$GMR_ROOT/general_motion_retargeting/__init__.py" \
    "$GMR_ROOT/assets/unitree_g1/g1_29dof.xml" \
    "$COMPAT_ROOT/deploy/onboard_deploy_wo_GMR/protocol.py"; do
    [[ -e "$path" ]] || { echo "[live-dryrun] missing: $path" >&2; exit 2; }
done
if [[ "$GMR_BACKEND" != "baseline" ]]; then
    for path in "$GMR_ROOT/native_gmr.py" "$GMR_ROOT/native_preprocess.py"; do
        [[ -e "$path" ]] || { echo "[live-dryrun] missing: $path" >&2; exit 2; }
    done
fi
if [[ "$GMR_BACKEND" == "native" && ! -e "$GMR_ROOT/libgmr_native.so" ]]; then
    echo "[live-dryrun] missing: $GMR_ROOT/libgmr_native.so" >&2
    exit 2
fi

if pgrep -af 'garment_udp_v2_reconnect_safe|publish_v51_reference|x2_deploy_onnx_ref' \
    | grep -v "$$" >/dev/null; then
    echo "[live-dryrun] another garment/adapter/deploy process is already running:" >&2
    pgrep -af 'garment_udp_v2_reconnect_safe|publish_v51_reference|x2_deploy_onnx_ref' >&2
    exit 3
fi
if ss -H -lun | awk '{print $5}' | grep -Eq '(^|:)51234$'; then
    echo "[live-dryrun] UDP :51234 is already in use" >&2
    exit 3
fi
if ss -H -ltn | awk '{print $4}' | grep -Eq '(^|:)5556$'; then
    echo "[live-dryrun] TCP :5556 is already in use" >&2
    exit 3
fi
bluetoothctl show 2>/dev/null | grep -q 'Powered: yes' || {
    echo "[live-dryrun] Bluetooth controller is not powered" >&2
    exit 4
}

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/garment_live_dryrun_$STAMP"
mkdir -p "$LOG_DIR"
CAPTURE="$LOG_DIR/hmcp_capture.jsonl"
ADAPTER_LOG="$LOG_DIR/adapter.log"
GARMENT_LOG="$LOG_DIR/garment.log"
DEPLOY_LOG="$LOG_DIR/deploy.log"
GRAPH_LOG="$LOG_DIR/hal_command_graph.log"
TPOSE_TRIGGER="$LOG_DIR/tpose.trigger"

adapter_pid=""
garment_pid=""
deploy_pid=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    # SIGINT makes the garment Python interpreter unwind while native worker
    # threads are still active, which ends in std::terminate/core dump.  TERM
    # the garment process directly; keep INT for the simpler adapter/deploy.
    if [[ -n "$garment_pid" ]] && kill -0 "$garment_pid" 2>/dev/null; then
        kill -TERM "$garment_pid" 2>/dev/null || true
    fi
    for pid in "$deploy_pid" "$adapter_pid"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "$deploy_pid" "$garment_pid" "$adapter_pid"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$deploy_pid" "$garment_pid" "$adapter_pid"; do
        [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
    done
    echo "[live-dryrun] logs: $LOG_DIR"
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo "[live-dryrun] adapter: HMCP udp://127.0.0.1:51234 -> ZMQ tcp://127.0.0.1:5556"
"$PYTHON_BIN" -u "$ADAPTER" \
    --source hmcp \
    --hmcp-bind 127.0.0.1 --hmcp-port 51234 \
    --host 127.0.0.1 --port 5556 --topic pose \
    --pose-scale 0.7 --diff-hz 50 --velocity-mode timestamp \
    --velocity-min-dt-s 0.005 --velocity-reset-gap-s 0.2 \
    --duration 0 --capture-jsonl "$CAPTURE" \
    >"$ADAPTER_LOG" 2>&1 &
adapter_pid=$!
sleep 1
kill -0 "$adapter_pid" 2>/dev/null || {
    echo "[live-dryrun] adapter failed" >&2
    tail -40 "$ADAPTER_LOG" >&2
    exit 5
}

echo "[live-dryrun] garments: jacket=$BLE_UP pants=$BLE_DOWN"
echo "[live-dryrun] reference offload: tcp://$REFERENCE_HOST:$REFERENCE_PORT epoch=$REFERENCE_EPOCH timeout=${REFERENCE_TIMEOUT_MS}ms startup_timeout=${REFERENCE_STARTUP_TIMEOUT_MS}ms startup_frames=$REFERENCE_STARTUP_FRAMES pipeline=$REFERENCE_PIPELINE"
if [[ "$EXPECT_REFERENCE_DISCONNECT" == "1" ]]; then
    echo "[live-dryrun] diagnostic: expecting reference disconnect and SAFE_IDLE watchdog trip"
fi
echo "[live-dryrun] source runtime: smpl_fast=$SMPL_FAST_DEVICE gmr=$GMR_BACKEND "\
"gmr_iter=$GMR_MAX_ITER torch_threads=$TORCH_THREADS root=$GMR_ROOT"
(
    cd "$SMARTWEAR_ROOT"
    export PYTHONPATH="$OFFLOAD_ROOT:$CUDA_OVERLAY:$COMPAT_ROOT:$GMR_ROOT:$MOCAP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$CUPTI_LIB:$CUDA_OVERLAY/onnxruntime/capi:$CUDA_OVERLAY/torch/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # The physics planner uses many tiny 24x3/75x75 operations. On this Orin,
    # PyTorch's default 8-way pools add more scheduling cost than computation.
    export OMP_NUM_THREADS="$TORCH_THREADS"
    export MKL_NUM_THREADS="$TORCH_THREADS"
    export OPENBLAS_NUM_THREADS="$TORCH_THREADS"
    export BNO085_BLE_AUTO_LECUP=0
    exec "$PYTHON_BIN" -u "$GARMENT_SENDER" \
        --input ble --addr-up "$BLE_UP" --addr-down "$BLE_DOWN" \
        --calibration-trigger-file "$TPOSE_TRIGGER" --fps 35 --no-viewer \
        --stream-mode udp --robot-ip 127.0.0.1 --udp-port 51234 \
        --hgpt-root "$COMPAT_ROOT" \
        --lfp-backend local-cuda \
        --smpl-converter fast --smpl-fast-device "$SMPL_FAST_DEVICE" \
        --gmr-backend "$GMR_BACKEND" --gmr-max-iter "$GMR_MAX_ITER" --gmr-damping 1.0 \
        --reference-backend soc2 \
        --reference-soc2-host "$REFERENCE_HOST" \
        --reference-soc2-port "$REFERENCE_PORT" \
        --reference-timeout-ms "$REFERENCE_TIMEOUT_MS" \
        --reference-startup-timeout-ms "$REFERENCE_STARTUP_TIMEOUT_MS" \
        --reference-startup-frames "$REFERENCE_STARTUP_FRAMES" \
        --reference-reset-timeout-s "$REFERENCE_RESET_TIMEOUT_S" \
        --reference-epoch "$REFERENCE_EPOCH" \
        "${reference_pipeline_args[@]}"
) >"$GARMENT_LOG" 2>&1 &
garment_pid=$!

echo "[live-dryrun] waiting for both garments; no T-Pose needed yet"
ble_deadline=$((SECONDS + BLE_READY_TIMEOUT))
while (( SECONDS < ble_deadline )); do
    kill -0 "$garment_pid" 2>/dev/null || {
        echo "[live-dryrun] garment process failed before calibration" >&2
        tail -100 "$GARMENT_LOG" >&2
        exit 6
    }
    if grep -qE 'Exception in thread|^Traceback \(most recent call last\)' "$GARMENT_LOG"; then
        echo "[live-dryrun] garment worker failed before calibration" >&2
        tail -100 "$GARMENT_LOG" >&2
        exit 6
    fi
    if grep -q 'TPOSE_GATE_READY' "$GARMENT_LOG"; then
        break
    fi
    sleep 1
done
if ! grep -q 'TPOSE_GATE_READY' "$GARMENT_LOG"; then
    echo "[live-dryrun] garments did not become ready within ${BLE_READY_TIMEOUT}s" >&2
    tail -100 "$GARMENT_LOG" >&2
    exit 7
fi

echo
echo "========== T-POSE READY =========="
echo "Both garments are connected and fresh. Hold a standard T-Pose now."
echo "Robot stays under official MC; this run cannot command HAL."
if [[ "${X2_TPOSE_CONFIRMED:-0}" != "1" ]]; then
    if [[ ! -t 0 ]]; then
        echo "[live-dryrun] T-Pose confirmation requires an interactive terminal" >&2
        exit 1
    fi
    printf "Type TPOSE to capture calibration: "
    answer=""
    while true; do
        if read -r -t 1 answer; then
            break
        fi
        if ! kill -0 "$garment_pid" 2>/dev/null; then
            echo
            echo "[live-dryrun] garment process failed while waiting for TPOSE" >&2
            tail -100 "$GARMENT_LOG" >&2
            exit 6
        fi
        if ! kill -0 "$adapter_pid" 2>/dev/null; then
            echo
            echo "[live-dryrun] adapter failed while waiting for TPOSE" >&2
            tail -100 "$ADAPTER_LOG" >&2
            exit 6
        fi
    done
    [[ "$answer" == "TPOSE" ]] || { echo "[live-dryrun] cancelled"; exit 1; }
fi
: >"$TPOSE_TRIGGER"

echo "[live-dryrun] waiting for $SOURCE_READY_FRAMES valid live HMCP frames"
deadline=$((SECONDS + SOURCE_READY_TIMEOUT))
while (( SECONDS < deadline )); do
    kill -0 "$garment_pid" 2>/dev/null || {
        echo "[live-dryrun] garment process failed" >&2
        tail -80 "$GARMENT_LOG" >&2
        exit 6
    }
    if grep -qE 'Exception in thread|^Traceback \(most recent call last\)' "$GARMENT_LOG"; then
        echo "[live-dryrun] garment worker failed" >&2
        tail -100 "$GARMENT_LOG" >&2
        exit 6
    fi
    capture_lines=0
    [[ ! -f "$CAPTURE" ]] || capture_lines="$(wc -l < "$CAPTURE" | tr -d ' ')"
    if (( capture_lines >= SOURCE_READY_FRAMES + 1 )); then
        break
    fi
    sleep 1
done
capture_lines=0
[[ ! -f "$CAPTURE" ]] || capture_lines="$(wc -l < "$CAPTURE" | tr -d ' ')"
if (( capture_lines < SOURCE_READY_FRAMES + 1 )); then
    echo "[live-dryrun] live source timeout: only $((capture_lines > 0 ? capture_lines - 1 : 0)) frames" >&2
    tail -80 "$GARMENT_LOG" >&2
    exit 7
fi
echo "[live-dryrun] live source ready: $((capture_lines - 1)) valid HMCP frames"

set +u
source /opt/ros/humble/setup.bash
source /agibot/data/home/agi/aimdk_ws_0_8_18/install/setup.bash
source "$SCRIPT_DIR/runtime_suspended/ws/install_gatecheck_corrected_20260829/setup.bash"
set -u

echo "[live-dryrun] starting corrected C++ Sonic for ${DRYRUN_SECONDS}s (HAL publishers disabled)"
LD_LIBRARY_PATH="$ORT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$DEPLOY_BIN" \
    --model "$MODEL" \
    --input-type zmq \
    --zmq-pose-host 127.0.0.1 --zmq-pose-port 5556 --zmq-pose-topic pose \
    --strict-reference-velocity \
    --zmq-warmup-body-frames 40 \
    --zmq-entry-max-age-s 0.5 --zmq-entry-min-fresh-s 0.8 \
    --pose-ref-stale-s 0.5 --pose-ref-min-fresh-s 1.0 \
    --dry-run --writer-hz 250 --autostart-after 3 \
    --max-duration "$DRYRUN_SECONDS" --log-dir "$LOG_DIR" \
    >"$DEPLOY_LOG" 2>&1 </dev/null &
deploy_pid=$!

sleep 2
kill -0 "$deploy_pid" 2>/dev/null || {
    echo "[live-dryrun] C++ deploy failed during startup" >&2
    tail -100 "$DEPLOY_LOG" >&2
    exit 8
}

for topic in leg waist arm head; do
    echo "===== /aima/hal/joint/$topic/command =====" >>"$GRAPH_LOG"
    ros2 topic info -v "/aima/hal/joint/$topic/command" >>"$GRAPH_LOG" 2>&1 || true
done
if grep -A8 -E 'Node name: (x2_deploy_onnx_ref|x2_sonic)' "$GRAPH_LOG" \
    | grep -q 'Endpoint type: PUBLISHER'; then
    echo "[live-dryrun] FAIL: dry-run unexpectedly created a HAL publisher" >&2
    exit 9
fi

starvation_verified=0
if [[ "$EXPECT_REFERENCE_DISCONNECT" == "1" ]]; then
    starvation_deadline=$((SECONDS + DRYRUN_SECONDS + 20))
    while kill -0 "$deploy_pid" 2>/dev/null && (( SECONDS < starvation_deadline )); do
        if grep -q 'CONTROL -> SAFE_IDLE: pose-ref starvation watchdog tripped' "$DEPLOY_LOG"; then
            starvation_verified=1
            break
        fi
        sleep 0.2
    done
    if [[ "$starvation_verified" != "1" ]]; then
        echo "[live-dryrun] FAIL: expected reference starvation did not reach SAFE_IDLE" >&2
        tail -100 "$GARMENT_LOG" >&2
        tail -120 "$DEPLOY_LOG" >&2
        exit 10
    fi
    if ! grep -qE 'Exception in thread|EOFError|ConnectionResetError|TimeoutError' "$GARMENT_LOG"; then
        echo "[live-dryrun] FAIL: SAFE_IDLE tripped without a recorded source failure" >&2
        tail -100 "$GARMENT_LOG" >&2
        exit 10
    fi
    frames_at_trip=0
    [[ ! -f "$CAPTURE" ]] || frames_at_trip="$(wc -l < "$CAPTURE" | tr -d ' ')"
    sleep 1
    frames_after_hold=0
    [[ ! -f "$CAPTURE" ]] || frames_after_hold="$(wc -l < "$CAPTURE" | tr -d ' ')"
    if (( frames_after_hold > frames_at_trip + 1 )); then
        echo "[live-dryrun] FAIL: HMCP continued after source failure: $frames_at_trip -> $frames_after_hold" >&2
        exit 10
    fi
    echo "[live-dryrun] PASS: reference disconnect stopped HMCP and watchdog entered SAFE_IDLE"
    kill -INT "$deploy_pid" 2>/dev/null || true
fi

set +e
wait "$deploy_pid"
deploy_rc=$?
set -e
deploy_pid=""
if (( deploy_rc != 0 )) && [[ "$EXPECT_REFERENCE_DISCONNECT" != "1" ]]; then
    echo "[live-dryrun] C++ deploy exited rc=$deploy_rc" >&2
    tail -120 "$DEPLOY_LOG" >&2
    exit "$deploy_rc"
fi

ticks=0
[[ ! -f "$LOG_DIR/tick.csv" ]] || ticks=$(( $(wc -l < "$LOG_DIR/tick.csv") - 1 ))
capture_lines="$(wc -l < "$CAPTURE" | tr -d ' ')"

read -r measured_frames source_hz source_p95_gap_ms source_max_gap_ms <<EOF
$("$PYTHON_BIN" - "$CAPTURE" <<'PY'
import json
import sys

timestamps = []
with open(sys.argv[1], "r", encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        timestamp = record.get("received_monotonic_ns")
        if timestamp is not None:
            timestamps.append(int(timestamp))

gaps_ms = [
    (current - previous) / 1e6
    for previous, current in zip(timestamps, timestamps[1:])
]
if not gaps_ms:
    print(len(timestamps), "0.00", "inf", "inf")
else:
    ordered = sorted(gaps_ms)
    p95_index = round(0.95 * (len(ordered) - 1))
    elapsed_s = (timestamps[-1] - timestamps[0]) / 1e9
    hz = (len(timestamps) - 1) / elapsed_s if elapsed_s > 0 else 0.0
    print(len(timestamps), f"{hz:.2f}", f"{ordered[p95_index]:.1f}", f"{max(gaps_ms):.1f}")
PY
)
EOF

echo "[live-dryrun] source: frames=$measured_frames rate=${source_hz}Hz p95_gap=${source_p95_gap_ms}ms max_gap=${source_max_gap_ms}ms"
if awk -v measured="$source_hz" -v target="$SOURCE_TARGET_HZ" 'BEGIN { exit !(measured >= target) }'; then
    echo "[live-dryrun] PASS: source throughput >= ${SOURCE_TARGET_HZ}Hz"
else
    echo "[live-dryrun] WARN: source throughput < ${SOURCE_TARGET_HZ}Hz; transport passed but powered garment gate remains closed" >&2
fi
echo "[live-dryrun] PASS: C++ ticks=$ticks HMCP_frames=$((capture_lines - 1))"
echo "[live-dryrun] PASS: no custom HAL command publisher detected; official MC stayed active"
"$PYTHON_BIN" "$SUMMARY" --capture "$CAPTURE" --deploy-log "$DEPLOY_LOG" || \
    echo "[live-dryrun] WARN: one-shot summary failed; raw logs remain available" >&2
