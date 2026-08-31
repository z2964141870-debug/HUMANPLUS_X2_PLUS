#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR="${X2_LEGACY_V2_DIR:-$HOME/projects/X2_sonic_legacy_v2_repro}"
PYTHON_BIN="${X2_SONIC_PYTHON:-$HOME/venvs/x2_sonic_cuda/bin/python}"
TORCH_OVERLAY="${X2_TORCH_GPU_OVERLAY:-$HOME/venvs/x2_sonic_torch_gpu_overlay}"
TORCH_DEPS="${X2_TORCH_GPU_DEPS:-$HOME/venvs/x2_sonic_torch_gpu_deps}"
SMARTWEAR_DIR="${X2_SMARTWEAR_V2_DIR:-$HOME/projects/smartwear_v2}"
MOCAP_DIR="${X2_MOCAP_DIR:-$HOME/mocap_teleop/x2_pc1_ble_teleop}"
BRIDGE_DIR="${X2_BRIDGE_DIR:-$HOME/projects/humanplus_x2_bridge}"
GMR_CPP_DIR="$REPRO_DIR/gmr_cpp_prototype"
GARMENT_SCRIPT="$REPRO_DIR/garment_udp_v2_reconnect_safe.py"
MODEL="$HOME/projects/X2_sonic_real/models/x2_sonic_frozen_g1core_lora_v2.onnx"
SCENE="$BRIDGE_DIR/x2_robot_assets/robot/scene.xml"
JACKET="${X2_BLE_UP:-FB:08:4D:3B:D6:06}"
PANTS="${X2_BLE_DOWN:-CC:D1:FA:DD:6D:B8}"
PORT="${X2_HMCP_PORT:-51234}"
SMPL_CONVERTER="${X2_SMPL_CONVERTER:-fast}"
SMPL_FAST_DEVICE="${X2_SMPL_FAST_DEVICE:-cpu}"
LFP_BACKEND="${X2_LFP_BACKEND:-local-cpu}"
LFP_SOC2_HOST="${X2_LFP_SOC2_HOST:-10.0.1.42}"
LFP_SOC2_PORT="${X2_LFP_SOC2_PORT:-51236}"
GMR_BACKEND="${X2_GMR_BACKEND:-native}"

if [[ "$SMPL_CONVERTER" != "fast" && "$SMPL_CONVERTER" != "smplx" ]]; then
  echo "[legacy-v2] X2_SMPL_CONVERTER must be fast or smplx" >&2
  exit 2
fi
if [[ "$SMPL_FAST_DEVICE" != "cpu" && "$SMPL_FAST_DEVICE" != "cuda" && "$SMPL_FAST_DEVICE" != "auto" ]]; then
  echo "[legacy-v2] X2_SMPL_FAST_DEVICE must be cpu, cuda, or auto" >&2
  exit 2
fi
if [[ "$LFP_BACKEND" != "local-cpu" && "$LFP_BACKEND" != "local-cuda" && "$LFP_BACKEND" != "soc2" ]]; then
  echo "[legacy-v2] X2_LFP_BACKEND must be local-cpu, local-cuda, or soc2" >&2
  exit 2
fi
if [[ "$GMR_BACKEND" != "baseline" && "$GMR_BACKEND" != "fixed" && "$GMR_BACKEND" != "native" ]]; then
  echo "[legacy-v2] X2_GMR_BACKEND must be baseline, fixed, or native" >&2
  exit 2
fi

required=(
  "$PYTHON_BIN"
  "$TORCH_OVERLAY/torch/__init__.py"
  "$TORCH_DEPS/nvtx/nvidia/nvtx/lib/libnvToolsExt.so.1"
  "$TORCH_DEPS/cupti/nvidia/cuda_cupti/lib/libcupti.so.12"
  "$TORCH_DEPS/cusparselt/cusparselt/lib/libcusparseLt.so.0"
  "$GARMENT_SCRIPT"
  "$REPRO_DIR/deploy/onboard_deploy_wo_GMR/protocol.py"
  "$SMARTWEAR_DIR/onnx_models/LFP_dense_taichi_ft.onnx"
  "$SMARTWEAR_DIR/onnx_models/TIC4Clothes_dense.onnx"
  "$BRIDGE_DIR/x2_realtime_closed_loop.py"
  "$BRIDGE_DIR/eval_official_sonic_x2.py"
  "$MODEL"
  "$SCENE"
)
if [[ "$SMPL_CONVERTER" == "smplx" ]]; then
  required+=("$REPRO_DIR/assets/body_models/smplx/SMPLX_NEUTRAL.npz")
fi
if [[ "$LFP_BACKEND" == "soc2" ]]; then
  required+=("$REPRO_DIR/lfp_remote.py")
fi
if [[ "$GMR_BACKEND" != "baseline" ]]; then
  required+=(
    "$GMR_CPP_DIR/native_gmr.py"
    "$GMR_CPP_DIR/native_preprocess.py"
    "$GMR_CPP_DIR/libgmr_native.so"
  )
fi
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[legacy-v2] missing required file: $path" >&2
    exit 2
  fi
done

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[legacy-v2] SHA256 mismatch: $path" >&2
    echo "[legacy-v2] expected=$expected actual=$actual" >&2
    exit 3
  fi
}

check_sha 76f6d90be49623f8a43eed2f8c548fe12f1294c3bb8f035ce82d8e6a4d8b51ae "$GARMENT_SCRIPT"
check_sha d51bc1bb25c72ccb8fcec266b66e86f1e1cbe5ef1afa07074b6156e107699419 "$REPRO_DIR/deploy/onboard_deploy_wo_GMR/protocol.py"
check_sha 183f29d64d8726793ff2e1462a7a05019dec303cc1ece345bcff9e8ac24d24d8 "$BRIDGE_DIR/x2_realtime_closed_loop.py"
check_sha 51631f3a9bd50419dd3f44ff9a1367dffb93aa36e815cd89c319ebd89802bf55 "$BRIDGE_DIR/eval_official_sonic_x2.py"
check_sha 8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9 "$MODEL"
check_sha 3b26151297d18cc5c49f78c3705b45a9cceda8b868e0f1a4ab49063fd22e7bf9 "$SCENE"

CUDA_LIB="/usr/local/cuda-12.6/targets/aarch64-linux/lib"
export PYTHONPATH="$TORCH_OVERLAY:$REPRO_DIR:$GMR_CPP_DIR:$SMARTWEAR_DIR:$MOCAP_DIR:$BRIDGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$TORCH_DEPS/nvtx/nvidia/nvtx/lib:$TORCH_DEPS/cupti/nvidia/cuda_cupti/lib:$TORCH_DEPS/cusparselt/cusparselt/lib:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GPU_INFO="$($PYTHON_BIN -c 'import torch; assert torch.cuda.is_available(); print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")')" || {
  echo "[legacy-v2] GPU PyTorch validation failed" >&2
  exit 6
}
mkdir -p "$REPRO_DIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
SIM_LOG="$REPRO_DIR/logs/sim_${STAMP}.log"
GARMENT_LOG="$REPRO_DIR/logs/garment_${STAMP}.log"
SOC2_LOG="$REPRO_DIR/logs/soc2_lfp_start_${STAMP}.log"

ACTIVE_LFP_BACKEND="$LFP_BACKEND"
if [[ "$LFP_BACKEND" == "soc2" ]]; then
  if ! aima robot run --local-target --soc 2 -- \
    /home/run/x2_aux_lfp/start_lfp_worker.sh >"$SOC2_LOG" 2>&1; then
    echo "[legacy-v2] SoC2 LFP startup failed; falling back to local ONNX" >&2
    echo "[legacy-v2] SoC2 startup log: $SOC2_LOG" >&2
    ACTIVE_LFP_BACKEND="local-cpu"
  fi
fi

SIM_PID=""
cleanup() {
  if [[ -n "$SIM_PID" ]]; then
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[legacy-v2] SIM/SHADOW ONLY: no HAL publisher, no MC transition, robot cannot move"
echo "[legacy-v2] V2 BLE -> TIC/LFP/GMR -> HMCP :$PORT -> Sonic -> official X2 MuJoCo"
echo "[legacy-v2] GPU runtime: $GPU_INFO"
echo "[legacy-v2] jacket=$JACKET pants=$PANTS pose_scale=0.7"
echo "[legacy-v2] SMPL converter=$SMPL_CONVERTER fast_device=$SMPL_FAST_DEVICE"
echo "[legacy-v2] LFP backend=$ACTIVE_LFP_BACKEND"
echo "[legacy-v2] GMR backend=$GMR_BACKEND"
echo "[legacy-v2] sim log: $SIM_LOG"
echo "[legacy-v2] garment log: $GARMENT_LOG"
echo "[legacy-v2] wait for the data-ready prompt; then hold T-Pose and press Enter once"

"$PYTHON_BIN" -u "$BRIDGE_DIR/x2_realtime_closed_loop.py" \
  --model "$MODEL" \
  --scene "$SCENE" \
  --listen "$PORT" \
  --pose-scale 0.7 \
  --backend sim \
  --headless > >(tee -a "$SIM_LOG") 2>&1 &
SIM_PID=$!

for _ in $(seq 1 60); do
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    wait "$SIM_PID" || true
    echo "[legacy-v2] MuJoCo shadow exited before HMCP startup" >&2
    exit 4
  fi
  if ss -lun 2>/dev/null | grep -q ":$PORT "; then
    break
  fi
  sleep 0.25
done

if ! ss -lun 2>/dev/null | grep -q ":$PORT "; then
  echo "[legacy-v2] HMCP receiver did not bind udp/:$PORT" >&2
  exit 5
fi

cd "$SMARTWEAR_DIR"
set +e
"$PYTHON_BIN" -u "$GARMENT_SCRIPT" \
  --input ble \
  --addr-up "$JACKET" \
  --addr-down "$PANTS" \
  --stream-mode udp \
  --robot-ip 127.0.0.1 \
  --udp-port "$PORT" \
  --hgpt-root "$REPRO_DIR" \
  --smplx-folder "$REPRO_DIR/assets/body_models" \
  --smpl-converter "$SMPL_CONVERTER" \
  --smpl-fast-device "$SMPL_FAST_DEVICE" \
  --lfp-backend "$ACTIVE_LFP_BACKEND" \
  --lfp-soc2-host "$LFP_SOC2_HOST" \
  --lfp-soc2-port "$LFP_SOC2_PORT" \
  --fps 35 \
  --gmr-max-iter 2 \
  --gmr-backend "$GMR_BACKEND" \
  --no-viewer 2>&1 | tee -a "$GARMENT_LOG"
STATUS="${PIPESTATUS[0]}"
set -e
exit "$STATUS"
