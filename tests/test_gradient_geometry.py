import torch

from dcpeft_gradient_geometry import (
    clipped_head_policy_losses,
    finite_summary,
    gradient_geometry,
)


def test_head_losses_exactly_reconstruct_combined_policy_loss() -> None:
    raw = torch.tensor(
        [
            [[1.0, -2.0], [3.0, 4.0], [9.0, 9.0]],
            [[-1.0, 2.0], [5.0, -6.0], [7.0, 8.0]],
        ]
    )
    clipped = raw + torch.tensor([0.5, -0.5])
    padding = torch.tensor([[False, False, True], [False, True, True]])
    heads, error, valid_steps = clipped_head_policy_losses(raw, clipped, padding)
    expected = torch.maximum(raw, clipped).sum(dim=-1)[~padding].mean()
    assert valid_steps == 3
    assert error < 1.0e-6
    assert torch.allclose(torch.stack(heads).sum(), expected, atol=1.0e-6, rtol=0.0)


def test_gradient_geometry_detects_conflict_and_alignment() -> None:
    conflict = gradient_geometry(
        {"loco": torch.tensor([1.0, 0.0]), "upper": torch.tensor([-1.0, 0.0])}
    )
    aligned = gradient_geometry(
        {"loco": torch.tensor([1.0, 1.0]), "upper": torch.tensor([2.0, 2.0])}
    )
    assert conflict["pairwise_cosines"]["loco__upper"] == -1.0
    assert conflict["pairwise_negative_fraction"] == 1.0
    assert abs(aligned["pairwise_cosines"]["loco__upper"] - 1.0) < 1.0e-12


def test_finite_summary_reports_negative_fraction() -> None:
    summary = finite_summary([-0.5, 0.0, 0.5, -0.1])
    assert summary["count"] == 4
    assert summary["negative_fraction"] == 0.5
    assert summary["min"] == -0.5
    assert summary["max"] == 0.5
