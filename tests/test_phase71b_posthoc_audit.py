import torch

from tools.retarget.audit_x2_phase71b_posthoc import (
    common_scale_components,
    paired_stability,
)


def test_float64_common_scale_components_close() -> None:
    generator = torch.Generator().manual_seed(71)
    total = torch.randn(400, 8, generator=generator)
    pitch = torch.randn(400, 8, generator=generator) * 0.1
    support = torch.randn(400, 8, generator=generator) * 0.1
    values = torch.randn(400, 8, generator=generator)
    _, _, closure = common_scale_components(
        total, values, pitch, support, dtype=torch.float64
    )
    assert closure["credit_relative_l2"] <= 1.0e-14
    assert closure["credit_max_abs"] <= 1.0e-13


def test_paired_stability_rejects_opposite_half_directions() -> None:
    direction = torch.ones(64, 3)
    direction[32:] *= -1.0
    metric = torch.ones(64, 3)
    summary = paired_stability(direction, metric, seed=710000)
    assert summary["direction_first_second_half_cosine"] <= -0.999999
    assert not summary["strict_stable"]
