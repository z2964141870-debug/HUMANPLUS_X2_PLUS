import io

import torch

from cwi_x2.phase_conditioned_knee_residual import PhaseConditionedKneeTargetResidual


def moving_observation(batch: int = 16) -> torch.Tensor:
    observation = torch.randn(batch, 93)
    observation[:, -4:] = torch.tensor([0.0, 1.0, 1.0, 1.0])
    return observation


def test_phase67_zero_initialization_is_exact_and_only_maps_two_knees():
    module = PhaseConditionedKneeTargetResidual()
    observation = moving_observation()
    output = module.target_offset_15d(observation)
    assert torch.equal(output, torch.zeros_like(output))
    assert output.shape == (16, 15)


def test_phase67_standing_is_zero_even_after_nonzero_head():
    module = PhaseConditionedKneeTargetResidual()
    torch.nn.init.constant_(module.head.weight, 1.0)
    torch.nn.init.constant_(module.head.bias, 1.0)
    observation = moving_observation()
    observation[:, -4:] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert torch.equal(module(observation), torch.zeros(16, 2))


def test_phase67_first_backward_reaches_only_zero_head_then_stays_bounded():
    module = PhaseConditionedKneeTargetResidual()
    observation = moving_observation()
    loss = -module(observation).sum()
    loss.backward()
    assert module.head.weight.grad is not None
    assert module.head.weight.grad.abs().sum() > 0
    assert module.head.bias.grad is not None
    assert module.head.bias.grad.abs().sum() > 0
    assert all(
        parameter.grad is not None and torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        for parameter in module.encoder.parameters()
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
    optimizer.step()
    assert module(observation).abs().max() <= 0.003 + 1.0e-8


def test_phase67_state_roundtrip_and_manifest_are_exact():
    module = PhaseConditionedKneeTargetResidual()
    observation = moving_observation()
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    buffer.seek(0)
    restored = PhaseConditionedKneeTargetResidual()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert torch.equal(module(observation), restored(observation))
    manifest = module.trainable_manifest()
    assert manifest["maximum_target_offset_rad"] == 0.003
    assert manifest["trainable_parameters"] == 4130
    assert manifest["trainable_names"] == [
        "encoder.0.bias", "encoder.0.weight", "encoder.2.bias", "encoder.2.weight",
        "head.bias", "head.weight",
    ]
