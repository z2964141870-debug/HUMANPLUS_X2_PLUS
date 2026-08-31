#!/usr/bin/env bash
set -Eeuo pipefail

# Source-only half of the proven live garment path.
#
#   X2 AX210 BLE -> local LFP -> 3588S reference offload -> HMCP UDP :51234
#   -> timestamp v5.1 adapter PUB :5555
#
# This script never starts the C++ deploy, never stops MC, and never creates a
# HAL command publisher. The guarded proxy and powered parent are separate.

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
OFFLOAD_ROOT="$SCRIPT_DIR/scripts/offload"

BLE_UP="${X2_BLE_UP:-F7:C6:1F:AB:37:E0}"
BLE_DOWN="${X2_BLE_DOWN:-DD:65:A4:4A:22:36}"
REFERENCE_HOST="${X2_REFERENCE_HOST:-10.0.1.42}"
REFERENCE_PORT="${X2_REFERENCE_PORT:-51237}"
REFERENCE_TIMEOUT_MS="${X2_REFERENCE_TIMEOUT_MS:-80}"
REFERENCE_STARTUP_TIMEOUT_MS="${X2_REFERENCE_STARTUP_TIMEOUT_MS:-1000}"
REFERENCE_STARTUP_FRAMES="${X2_REFERENCE_STARTUP_FRAMES:-50}"
REFERENCE_RESET_TIMEOUT_S="${X2_REFERENCE_RESET_TIMEOUT_S:-10}"
REFERENCE_EPOCH="${X2_REFERENCE_EPOCH:-$(date +%s)}"
BLE_READY_TIMEOUT="${X2_BLE_READY_TIMEOUT:-180}"
SOURCE_READY_TIMEOUT="${X2_SOURCE_READY_TIMEOUT:-120}"
SOURCE_READY_FRAMES="${X2_SOURCE_READY_FRAMES:-80}"
SOURCE_TARGET_HZ="${X2_SOURCE_TARGET_HZ:-30}"

for integer_name in REFERENCE_STARTUP_FRAMES SOURCE_READY_FRAMES; do
    integer_value="${!integer_name}"
    [[ "$integer_value" =~ ^[1-9][0-9]*$ ]] || {
        echo "[garment-source] invalid $integer_name=$integer_value" >&2
        exit 2
    }
done

for path in \
    "$PYTHON_BIN" \
    "$GARMENT_SENDER" \
    "$ADAPTER" \
    "$OFFLOAD_ROOT/reference_offload_client.py" \
    "$OFFLOAD_ROOT/reference_offload_protocol.py" \
    "$SMARTWEAR_ROOT/onnx_models/LFP_dense_taichi_ft.onnx" \
    "$SMARTWEAR_ROOT/onnx_models/TIC4Clothes_dense.onnx" \
    "$GMR_ROOT/general_motion_retargeting/__init__.py" \
    "$GMR_ROOT/native_gmr.py" \
    "$GMR_ROOT/native_preprocess.py" \
    "$GMR_ROOT/libgmr_native.so" \
    "$GMR_ROOT/assets/unitree_g1/g1_29dof.xml" \
    "$COMPAT_ROOT/deploy/onboard_deploy_wo_GMR/protocol.py"; do
    [[ -e "$path" ]] || { echo "[garment-source] missing: $path" >&2; exit 2; }
done

if pgrep -af 'garment_udp_v2_reference_offload|garment_udp_v2_reconnect_safe|publish_v51_reference' \
    | grep -v "$$" >/dev/null; then
    echo "[garment-source] another garment or adapter process is running:" >&2
    pgrep -af 'garment_udp_v2_reference_offload|garment_udp_v2_reconnect_safe|publish_v51_reference' >&2
    exit 3
fi
if ss -H -lun | awk '{print $5}' | grep -Eq '(^|:)51234$'; then
    echo "[garment-source] UDP :51234 is already in use" >&2
    exit 3
fi
if ss -H -ltn | awk '{print $4}' | grep -Eq '(^|:)5555$'; then
    echo "[garment-source] TCP :5555 is already in use" >&2
    exit 3
fi
bluetoothctl show 2>/dev/null | grep -q 'Powered: yes' || {
    echo "[garment-source] Bluetooth controller is not powered" >&2
    exit 4
}

"$PYTHON_BIN" - "$REFERENCE_HOST" "$REFERENCE_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1.0):
        pass
except OSError as exc:
    raise SystemExit(f"[garment-source] reference offload is unreachable at {host}:{port}: {exc}")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs/garment_offload_source_$STAMP"
mkdir -p "$LOG_DIR"
CAPTURE="$LOG_DIR/hmcp_capture.jsonl"
ADAPTER_LOG="$LOG_DIR/adapter.log"
GARMENT_LOG="$LOG_DIR/garment.log"
TPOSE_TRIGGER="$LOG_DIR/tpose.trigger"
STATUS_FILE="$LOG_DIR/source_status.json"

adapter_pid=""
garment_pid=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "$garment_pid" ]] && kill -0 "$garment_pid" 2>/dev/null; then
        kill -TERM "$garment_pid" 2>/dev/null || true
    fi
    if [[ -n "$adapter_pid" ]] && kill -0 "$adapter_pid" 2>/dev/null; then
        kill -INT "$adapter_pid" 2>/dev/null || true
    fi
    sleep 1
    for pid in "$garment_pid" "$adapter_pid"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$garment_pid" "$adapter_pid"; do
        [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
    done
    echo "[garment-source] stopped; logs: $LOG_DIR"
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo "[garment-source] adapter HMCP udp://127.0.0.1:51234 -> raw ZMQ tcp://127.0.0.1:5555"
"$PYTHON_BIN" -u "$ADAPTER" \
    --source hmcp \
    --hmcp-bind 127.0.0.1 --hmcp-port 51234 \
    --host 127.0.0.1 --port 5555 --topic pose \
    --pose-scale 0.7 --diff-hz 50 --velocity-mode timestamp \
    --velocity-min-dt-s 0.005 --velocity-reset-gap-s 0.2 \
    --duration 0 --capture-jsonl "$CAPTURE" \
    >"$ADAPTER_LOG" 2>&1 &
adapter_pid=$!
sleep 1
kill -0 "$adapter_pid" 2>/dev/null || {
    echo "[garment-source] adapter failed" >&2
    tail -80 "$ADAPTER_LOG" >&2
    exit 5
}

echo "[garment-source] garments jacket=$BLE_UP pants=$BLE_DOWN"
echo "[garment-source] offload tcp://$REFERENCE_HOST:$REFERENCE_PORT epoch=$REFERENCE_EPOCH pipeline=1"
echo "[garment-source] frozen runtime smpl_fast=cpu gmr=native iter=2 threads=1"
(
    cd "$SMARTWEAR_ROOT"
    export PYTHONPATH="$OFFLOAD_ROOT:$CUDA_OVERLAY:$COMPAT_ROOT:$GMR_ROOT:$MOCAP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$CUPTI_LIB:$CUDA_OVERLAY/onnxruntime/capi:$CUDA_OVERLAY/torch/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BNO085_BLE_AUTO_LECUP=0
    exec "$PYTHON_BIN" -u "$GARMENT_SENDER" \
        --input ble --addr-up "$BLE_UP" --addr-down "$BLE_DOWN" \
        --calibration-trigger-file "$TPOSE_TRIGGER" --fps 35 --no-viewer \
        --stream-mode udp --robot-ip 127.0.0.1 --udp-port 51234 \
        --hgpt-root "$COMPAT_ROOT" \
        --lfp-backend local-cuda \
        --smpl-converter fast --smpl-fast-device cpu \
        --gmr-backend native --gmr-max-iter 2 --gmr-damping 1.0 \
        --reference-backend soc2 \
        --reference-soc2-host "$REFERENCE_HOST" \
        --reference-soc2-port "$REFERENCE_PORT" \
        --reference-timeout-ms "$REFERENCE_TIMEOUT_MS" \
        --reference-startup-timeout-ms "$REFERENCE_STARTUP_TIMEOUT_MS" \
        --reference-startup-frames "$REFERENCE_STARTUP_FRAMES" \
        --reference-reset-timeout-s "$REFERENCE_RESET_TIMEOUT_S" \
        --reference-epoch "$REFERENCE_EPOCH" \
        --reference-pipeline
) >"$GARMENT_LOG" 2>&1 &
garment_pid=$!

echo "[garment-source] waiting for both garments; do not T-Pose yet"
ble_deadline=$((SECONDS + BLE_READY_TIMEOUT))
while (( SECONDS < ble_deadline )); do
    kill -0 "$garment_pid" 2>/dev/null || {
        echo "[garment-source] garment process failed before calibration" >&2
        tail -120 "$GARMENT_LOG" >&2
        exit 6
    }
    if grep -qE 'Exception in thread|^Traceback \(most recent call last\)' "$GARMENT_LOG"; then
        echo "[garment-source] garment worker failed before calibration" >&2
        tail -120 "$GARMENT_LOG" >&2
        exit 6
    fi
    grep -q 'TPOSE_GATE_READY' "$GARMENT_LOG" && break
    sleep 1
done
grep -q 'TPOSE_GATE_READY' "$GARMENT_LOG" || {
    echo "[garment-source] garments did not become ready in ${BLE_READY_TIMEOUT}s" >&2
    tail -120 "$GARMENT_LOG" >&2
    exit 7
}

echo
echo "========== T-POSE READY =========="
echo "Both garments are connected and fresh. Hold a standard T-Pose now."
echo "This source-only process cannot command MC or HAL."
if [[ "${X2_TPOSE_CONFIRMED:-0}" != "1" ]]; then
    [[ -t 0 ]] || { echo "[garment-source] interactive T-Pose confirmation required" >&2; exit 1; }
    printf "Type TPOSE to capture calibration: "
    answer=""
    while true; do
        read -r -t 1 answer && break
        kill -0 "$garment_pid" 2>/dev/null || {
            echo
            echo "[garment-source] garment process failed while waiting for TPOSE" >&2
            tail -120 "$GARMENT_LOG" >&2
            exit 6
        }
    done
    [[ "$answer" == "TPOSE" ]] || { echo "[garment-source] cancelled"; exit 1; }
fi
: >"$TPOSE_TRIGGER"

echo "[garment-source] waiting for $SOURCE_READY_FRAMES valid HMCP frames"
source_deadline=$((SECONDS + SOURCE_READY_TIMEOUT))
while (( SECONDS < source_deadline )); do
    kill -0 "$garment_pid" 2>/dev/null || {
        echo "[garment-source] garment process failed" >&2
        tail -120 "$GARMENT_LOG" >&2
        exit 6
    }
    kill -0 "$adapter_pid" 2>/dev/null || {
        echo "[garment-source] adapter failed" >&2
        tail -120 "$ADAPTER_LOG" >&2
        exit 6
    }
    capture_lines=0
    [[ ! -f "$CAPTURE" ]] || capture_lines="$(wc -l < "$CAPTURE" | tr -d ' ')"
    (( capture_lines >= SOURCE_READY_FRAMES + 1 )) && break
    sleep 1
done

capture_lines=0
[[ ! -f "$CAPTURE" ]] || capture_lines="$(wc -l < "$CAPTURE" | tr -d ' ')"
if (( capture_lines < SOURCE_READY_FRAMES + 1 )); then
    echo "[garment-source] source timeout: only $((capture_lines > 0 ? capture_lines - 1 : 0)) frames" >&2
    tail -120 "$GARMENT_LOG" >&2
    exit 7
fi

measure_source() {
    "$PYTHON_BIN" - "$CAPTURE" "$STATUS_FILE" "$SOURCE_TARGET_HZ" <<'PY'
import json
import os
import sys
import time

capture, status_path, target_hz = sys.argv[1], sys.argv[2], float(sys.argv[3])
timestamps = []
with open(capture, "r", encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        value = record.get("received_monotonic_ns")
        if value is not None:
            timestamps.append(int(value))
recent = timestamps[-180:]
if len(recent) < 2:
    rate = 0.0
    p95_ms = None
    max_ms = None
else:
    gaps = [(b - a) / 1e6 for a, b in zip(recent, recent[1:])]
    elapsed = (recent[-1] - recent[0]) / 1e9
    rate = (len(recent) - 1) / elapsed if elapsed > 0.0 else 0.0
    ordered = sorted(gaps)
    p95_ms = ordered[round(0.95 * (len(ordered) - 1))]
    max_ms = max(gaps)
payload = {
    "schema": "x2-garment-offload-source-status-v1",
    "updated_wall_s": time.time(),
    "pid": os.getppid(),
    "frames": len(timestamps),
    "recent_rate_hz": rate,
    "recent_p95_gap_ms": p95_ms,
    "recent_max_gap_ms": max_ms,
    "target_hz": target_hz,
    "ready": rate >= target_hz,
    "raw_zmq_endpoint": "tcp://127.0.0.1:5555",
}
temporary = status_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
os.replace(temporary, status_path)
print(f"{rate:.2f} {p95_ms if p95_ms is not None else float('inf'):.1f} "
      f"{max_ms if max_ms is not None else float('inf'):.1f} {len(timestamps)}")
PY
}

read -r source_hz source_p95_ms source_max_ms measured_frames <<<"$(measure_source)"
if ! awk -v measured="$source_hz" -v target="$SOURCE_TARGET_HZ" 'BEGIN { exit !(measured >= target) }'; then
    echo "[garment-source] FAIL: ${source_hz}Hz < ${SOURCE_TARGET_HZ}Hz; guarded powered path remains closed" >&2
    exit 8
fi
echo "[garment-source] SOURCE_READY frames=$measured_frames rate=${source_hz}Hz p95=${source_p95_ms}ms max=${source_max_ms}ms"
echo "[garment-source] raw pose is on :5555 only; no C++ deploy or HAL writer was started"
echo "[garment-source] logs: $LOG_DIR"

while true; do
    sleep 5
    kill -0 "$garment_pid" 2>/dev/null || {
        echo "[garment-source] garment process exited; stopping raw reference" >&2
        tail -120 "$GARMENT_LOG" >&2
        exit 9
    }
    kill -0 "$adapter_pid" 2>/dev/null || {
        echo "[garment-source] adapter exited; stopping raw reference" >&2
        tail -120 "$ADAPTER_LOG" >&2
        exit 9
    }
    if grep -qE 'Exception in thread|^Traceback \(most recent call last\)' "$GARMENT_LOG"; then
        echo "[garment-source] garment worker faulted; stopping raw reference" >&2
        tail -120 "$GARMENT_LOG" >&2
        exit 9
    fi
    read -r source_hz source_p95_ms source_max_ms measured_frames <<<"$(measure_source)"
    echo "[garment-source] live frames=$measured_frames rate=${source_hz}Hz p95=${source_p95_ms}ms max=${source_max_ms}ms"
done
