from __future__ import annotations

import ctypes
import platform
from pathlib import Path

import numpy as np


_FLOAT64_PTR = np.ctypeslib.ndpointer(
    dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED")
)


def _library_path() -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return Path(__file__).resolve().parent / f"libgmr_native{suffix}"


class NativeFramePreprocessor:
    def __init__(self, body_names, scales, position_offsets, rotation_offsets, root_name):
        self.body_names = tuple(body_names)
        self.root_index = self.body_names.index(root_name)
        self.scales = np.ascontiguousarray(
            [scales[name] for name in self.body_names], dtype=np.float64
        )
        self.position_offsets = np.ascontiguousarray(
            [position_offsets[name] for name in self.body_names], dtype=np.float64
        ).reshape(-1)
        self.rotation_offsets = np.ascontiguousarray(
            [rotation_offsets[name].as_quat(scalar_first=True) for name in self.body_names],
            dtype=np.float64,
        ).reshape(-1)
        self.output_positions = np.empty((len(self.body_names), 3), dtype=np.float64)
        self.output_quaternions = np.empty((len(self.body_names), 4), dtype=np.float64)

        library_path = _library_path()
        if not library_path.is_file():
            raise FileNotFoundError(f"Native library is missing; run build.sh first: {library_path}")
        self.library = ctypes.CDLL(str(library_path))
        self.function = self.library.gmr_preprocess
        self.function.argtypes = [
            _FLOAT64_PTR,
            _FLOAT64_PTR,
            _FLOAT64_PTR,
            _FLOAT64_PTR,
            _FLOAT64_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            _FLOAT64_PTR,
            _FLOAT64_PTR,
        ]
        self.function.restype = ctypes.c_int

    def __call__(self, human_data, ground_offset=0.0):
        positions = np.ascontiguousarray(
            [human_data[name][0] for name in self.body_names], dtype=np.float64
        ).reshape(-1)
        quaternions = np.ascontiguousarray(
            [human_data[name][1] for name in self.body_names], dtype=np.float64
        ).reshape(-1)
        status = self.function(
            positions,
            quaternions,
            self.scales,
            self.position_offsets,
            self.rotation_offsets,
            len(self.body_names),
            self.root_index,
            float(ground_offset),
            self.output_positions.reshape(-1),
            self.output_quaternions.reshape(-1),
        )
        if status != 0:
            raise RuntimeError(f"gmr_preprocess failed with status {status}")
        return {
            name: (self.output_positions[index].copy(), self.output_quaternions[index].copy())
            for index, name in enumerate(self.body_names)
        }
