#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from replay_x2_sonic_adapter import replay_file
from x2_sonic_input_adapter import DEFAULT_ANGLES_MJ, X2SonicAdapterConfig


class ReplayTest(unittest.TestCase):
    def _write(self, path: Path, records) -> None:
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_replay_writes_output_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "adapted.jsonl"
            manifest_path = root / "manifest.json"
            self._write(source, [
                {"timestamp_s": 0.0, "joint_pos": (DEFAULT_ANGLES_MJ + 0.2).tolist(), "root_quat": [1, 0, 0, 0]},
                {"timestamp_s": 0.02, "joint_pos": (DEFAULT_ANGLES_MJ + 0.4).tolist(), "root_quat": [1, 0, 0, 0]},
            ])
            manifest = replay_file(
                source,
                output,
                manifest_path,
                X2SonicAdapterConfig(root_tilt_scale=0.0, pose_scale=0.5),
            )
            self.assertEqual(manifest["schema"], "x2_sonic_garment_replay_jsonl_v1")
            self.assertEqual(manifest["output"]["records"], 2)
            lines = [json.loads(line) for line in output.read_text().splitlines()]
            np.testing.assert_allclose(lines[0]["joint_pos"], (DEFAULT_ANGLES_MJ + 0.1).tolist())
            self.assertEqual(lines[0]["root_quat"], [1.0, 0.0, 0.0, 0.0])
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["garment_mapping"]["status"], "pending")

    def test_non_monotonic_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            self._write(source, [
                {"timestamp_s": 1.0, "joint_pos": DEFAULT_ANGLES_MJ.tolist(), "root_quat": [1, 0, 0, 0]},
                {"timestamp_s": 0.5, "joint_pos": DEFAULT_ANGLES_MJ.tolist(), "root_quat": [1, 0, 0, 0]},
            ])
            with self.assertRaisesRegex(ValueError, "not monotonic"):
                replay_file(
                    source,
                    root / "adapted.jsonl",
                    root / "manifest.json",
                    X2SonicAdapterConfig(),
                )


if __name__ == "__main__":
    unittest.main()
