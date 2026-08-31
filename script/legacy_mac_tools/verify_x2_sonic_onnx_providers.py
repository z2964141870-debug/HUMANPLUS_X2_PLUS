#!/usr/bin/env python3
"""Verify CUDA loading and CPU/GPU numerical parity for X2-Sonic ONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eval_official_sonic_x2 import OBS_DIM, N_JOINTS, preload_onnx_cuda_runtime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("samples must be at least 2")

    preloaded, preload_error = preload_onnx_cuda_runtime()
    gpu = ort.InferenceSession(
        str(args.model),
        providers=[
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ],
    )
    cpu = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    gpu_providers = gpu.get_providers()
    if not gpu_providers or gpu_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"CUDA session unavailable: providers={gpu_providers}, "
            f"preload_error={preload_error}"
        )

    input_info = gpu.get_inputs()[0]
    output_info = gpu.get_outputs()[0]
    if input_info.shape[1] != OBS_DIM or output_info.shape[1] != N_JOINTS:
        raise RuntimeError(
            f"unexpected ONNX contract: {input_info.shape} -> {output_info.shape}"
        )
    rng = np.random.default_rng(args.seed)
    observations = rng.normal(0.0, 0.25, (args.samples, OBS_DIM)).astype(np.float32)
    observations[0] = 0.0
    observations[1] = 1.0
    gpu_output = gpu.run([output_info.name], {input_info.name: observations})[0]
    cpu_output = cpu.run([output_info.name], {input_info.name: observations})[0]
    difference = np.asarray(gpu_output, dtype=np.float64) - np.asarray(cpu_output, dtype=np.float64)
    finite = bool(np.all(np.isfinite(gpu_output)) and np.all(np.isfinite(cpu_output)))
    allclose = bool(np.allclose(gpu_output, cpu_output, atol=args.atol, rtol=args.rtol))
    report = {
        "schema": "x2_sonic_onnx_provider_parity_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline_only": True,
        "hardware_touched": False,
        "host": platform.node(),
        "model": {
            "path": str(args.model.resolve()),
            "sha256": sha256(args.model),
            "bytes": args.model.stat().st_size,
            "input_shape": list(input_info.shape),
            "output_shape": list(output_info.shape),
        },
        "runtime": {
            "onnxruntime_gpu": package_version("onnxruntime-gpu"),
            "nvidia_cublas_cu12": package_version("nvidia-cublas-cu12"),
            "nvidia_cudnn_cu12": package_version("nvidia-cudnn-cu12"),
            "nvidia_cuda_runtime_cu12": package_version("nvidia-cuda-runtime-cu12"),
            "available_providers": ort.get_available_providers(),
            "cuda_session_providers": gpu_providers,
            "cuda_provider_options": gpu.get_provider_options().get(
                "CUDAExecutionProvider", {}
            ),
            "cpu_session_providers": cpu.get_providers(),
            "cuda_runtime_preloaded": preloaded,
            "cuda_runtime_preload_error": preload_error,
        },
        "input": {"samples": args.samples, "seed": args.seed, "dtype": "float32"},
        "comparison": {
            "finite": finite,
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "atol": args.atol,
            "rtol": args.rtol,
            "allclose": allclose,
        },
        "pass": finite and allclose,
        "interpretation": (
            "Provider loading and feed-forward numerical parity only; this does not "
            "establish closed-loop dynamics or hardware readiness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
