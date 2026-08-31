#!/usr/bin/env python3

import unittest
from unittest import mock

import eval_official_sonic_x2 as target


class PreloadOnnxCudaRuntimeTest(unittest.TestCase):
    def test_provider_unavailable(self):
        with mock.patch.object(target.ort, "get_available_providers", return_value=["CPUExecutionProvider"]):
            loaded, error = target.preload_onnx_cuda_runtime()
        self.assertFalse(loaded)
        self.assertIn("not advertised", error)

    def test_preload_success(self):
        with (
            mock.patch.object(target.ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
            mock.patch.object(target.ort, "preload_dlls") as preload,
        ):
            loaded, error = target.preload_onnx_cuda_runtime()
        self.assertTrue(loaded)
        self.assertIsNone(error)
        preload.assert_called_once_with()

    def test_preload_failure_is_reported(self):
        with (
            mock.patch.object(target.ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
            mock.patch.object(target.ort, "preload_dlls", side_effect=RuntimeError("missing runtime")),
        ):
            loaded, error = target.preload_onnx_cuda_runtime()
        self.assertFalse(loaded)
        self.assertEqual(error, "RuntimeError: missing runtime")


if __name__ == "__main__":
    unittest.main()
