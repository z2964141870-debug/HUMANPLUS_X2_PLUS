#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="/home/run/x2_aux_lfp"
PYTHONPATH_DIR="/home/run/x2_aux_ort"
MODEL="$WORK_DIR/LFP_dense_taichi_ft.onnx"
WORKER="$WORK_DIR/lfp_worker.py"
PID_FILE="$WORK_DIR/lfp_worker.pid"
LOG_FILE="$WORK_DIR/lfp_worker.log"
PORT="${X2_SOC2_LFP_PORT:-51236}"
CLIENT_TIMEOUT="${X2_SOC2_LFP_CLIENT_TIMEOUT:-1.0}"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    if [[ -f "$LOG_FILE" ]] && grep -q "READY protocol=LFP2" "$LOG_FILE"; then
      echo "[soc2-lfp] already running pid=$PID port=$PORT protocol=LFP2"
      exit 0
    fi
    echo "[soc2-lfp] stopping stale pre-LFP2 worker pid=$PID"
    kill "$PID"
    for _ in $(seq 1 30); do
      if ! kill -0 "$PID" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo "[soc2-lfp] stale worker did not stop; refusing to start a second worker" >&2
      exit 1
    fi
  fi
fi

nohup env PYTHONPATH="$PYTHONPATH_DIR" python3 -u "$WORKER" \
  --model "$MODEL" \
  --bind 10.0.1.42 \
  --port "$PORT" \
  --threads 4 \
  --client-timeout "$CLIENT_TIMEOUT" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

for _ in $(seq 1 50); do
  if ! kill -0 "$PID" 2>/dev/null; then
    tail -20 "$LOG_FILE" >&2
    exit 1
  fi
  if grep -q "READY protocol=LFP2" "$LOG_FILE"; then
    echo "[soc2-lfp] started pid=$PID port=$PORT"
    exit 0
  fi
  sleep 0.1
done

echo "[soc2-lfp] startup timed out; see $LOG_FILE" >&2
exit 1
