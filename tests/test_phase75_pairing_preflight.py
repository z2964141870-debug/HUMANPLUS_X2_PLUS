from pathlib import Path

import pytest
import torch

from cwi_x2.phase75_pairing_preflight import copy_pair_rows, pair_diagnostics, paired_lane_ids, select_rows


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_value_accepts_cpu_indices() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    donors, recipients = paired_lane_ids(0, 4)
    copy_pair_rows(value, donors, recipients)
    assert pair_diagnostics(value, donors, recipients)["passed"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed CPU/CUDA evidence test requires CUDA")
def test_mixed_device_indices_are_normalized_to_value_device() -> None:
    cpu_value = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    cuda_donors, cuda_recipients = paired_lane_ids(0, 4, device="cuda:0")
    assert select_rows(cpu_value, cuda_donors).device.type == "cpu"
    copy_pair_rows(cpu_value, cuda_donors, cuda_recipients)
    assert pair_diagnostics(cpu_value, cuda_donors, cuda_recipients)["passed"] is True

    cuda_value = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(8, 2)
    cpu_donors, cpu_recipients = paired_lane_ids(0, 4, device="cpu")
    copy_pair_rows(cuda_value, cpu_donors, cpu_recipients)
    assert pair_diagnostics(cuda_value, cpu_donors, cpu_recipients)["passed"] is True


def test_axis1_buffer_is_copied_through_batch_first_view() -> None:
    value = torch.arange(48, dtype=torch.float32).reshape(3, 8, 2)
    donors, recipients = paired_lane_ids(0, 4)
    batch_first = value.transpose(0, 1)
    copy_pair_rows(batch_first, donors, recipients)
    assert pair_diagnostics(batch_first, donors, recipients)["passed"] is True


def test_dynamic_runner_transform_is_narrow_and_device_safe() -> None:
    text = (ROOT / "scripts/run_x2_phase75_pairing_preflight.py").read_text()
    assert "EXPECTED_PHASE74_SHA256" in text
    assert 'source.count(old)' in text
    assert 'tensor_hash(select_rows(field_value, donors))' in text
    assert 'copy_pair_rows(value.transpose(0, 1), donors, recipients)' in text
    assert 'mixed_device_evidence_path_exercised' in text
    assert 'phase76_shadow_preregistration_unlocked' in text


def test_phase75_remains_initial_only() -> None:
    frozen = (ROOT / "scripts/run_x2_phase74_pairing_preflight.py").read_text()
    assert "env.step(" not in frozen
    assert "reward_manager.compute" not in frozen
    assert "torch.optim" not in frozen
    assert ".backward(" not in frozen

