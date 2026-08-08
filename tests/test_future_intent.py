from __future__ import annotations

import torch

from cwi_x2.future_intent_actor_critic import (
    BASE_ACTOR_OBS_DIM,
    DYNAMIC_INTENT_DIM,
    LOCOMOTION_INTENT_DIM,
    NUM_COORDINATION_MODES,
    NUM_LOWER_ACTIONS,
    RESPONSE_CONTEXT_DIM,
    UPPER_INTENT_DIM,
    FutureIntentActorCritic,
    build_coordination_basis,
    lower_response_context,
)


def _model(
    mode: str = "future",
    *,
    response_adapter: bool = False,
    locomotion_intent_only: bool = False,
    transition_adapter: bool = False,
) -> FutureIntentActorCritic:
    obs = {
        "policy": torch.zeros(2, BASE_ACTOR_OBS_DIM + DYNAMIC_INTENT_DIM),
        "critic": torch.zeros(2, BASE_ACTOR_OBS_DIM),
    }
    return FutureIntentActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=NUM_LOWER_ACTIONS,
        adapter_mode=mode,
        response_adapter_enabled=response_adapter,
        locomotion_intent_only=locomotion_intent_only,
        transition_adapter_enabled=transition_adapter,
    )


def test_coordination_basis_is_orthonormal_and_sparse():
    basis = build_coordination_basis()
    assert basis.shape == (NUM_COORDINATION_MODES, NUM_LOWER_ACTIONS)
    torch.testing.assert_close(
        basis @ basis.T,
        torch.eye(NUM_COORDINATION_MODES),
    )
    # Knee and ankle columns are never directly modified.
    assert torch.count_nonzero(basis[:, [3, 4, 5, 9, 10, 11, 13]]) == 0


def test_lower_response_context_is_bounded_and_deployable():
    base = 100.0 * torch.randn(4, BASE_ACTOR_OBS_DIM)
    context = lower_response_context(base)
    assert context.shape == (4, RESPONSE_CONTEXT_DIM)
    assert float(torch.max(torch.abs(context[..., : 2 * NUM_LOWER_ACTIONS]))) <= 1.0
    torch.testing.assert_close(
        context[..., -NUM_LOWER_ACTIONS:],
        base[..., 74:89],
    )


def test_zero_initialized_response_branch_preserves_trained_future_branch():
    torch.manual_seed(5)
    legacy = _model("future")
    torch.nn.init.normal_(legacy.coordination_adapter[-1].weight, std=0.2)
    torch.nn.init.normal_(legacy.coordination_adapter[-1].bias, std=0.2)
    response = _model("future", response_adapter=True)
    response.load_state_dict(legacy.state_dict(), strict=True)
    base = torch.randn(4, BASE_ACTOR_OBS_DIM)
    base[:, 9] = 0.3
    intent = 0.03 * torch.randn(4, DYNAMIC_INTENT_DIM)
    observation = torch.cat((base, intent), dim=-1)
    torch.testing.assert_close(
        response._mean_from_actor_observation(observation),
        legacy._mean_from_actor_observation(observation),
        rtol=0.0,
        atol=0.0,
    )
    assert not any(p.requires_grad for p in response.coordination_adapter.parameters())
    assert all(p.requires_grad for p in response.response_adapter.parameters())


def test_response_branch_can_act_without_upper_intent_but_is_bounded():
    torch.manual_seed(6)
    model = _model("future", response_adapter=True)
    torch.nn.init.normal_(model.response_adapter[-1].weight, std=10.0)
    torch.nn.init.normal_(model.response_adapter[-1].bias, std=10.0)
    base = torch.randn(3, BASE_ACTOR_OBS_DIM)
    base[:, 9] = 0.3
    intent = torch.zeros(3, DYNAMIC_INTENT_DIM)
    output = model._mean_from_actor_observation(torch.cat((base, intent), dim=-1))
    residual = output - model.actor(base)
    assert float(torch.max(torch.abs(residual)).detach()) <= 0.050001
    torch.testing.assert_close(
        residual[:, [3, 4, 5, 9, 10, 11, 13]],
        torch.zeros_like(residual[:, [3, 4, 5, 9, 10, 11, 13]]),
    )
    torch.testing.assert_close(
        residual[:, [2, 8, 12]],
        torch.zeros_like(residual[:, [2, 8, 12]]),
    )


def test_locomotion_only_gradient_mask_protects_existing_response_columns():
    torch.manual_seed(8)
    model = _model(
        "future",
        response_adapter=True,
        locomotion_intent_only=True,
    )
    torch.nn.init.normal_(model.response_adapter[-1].weight, std=0.2)
    base = torch.randn(4, BASE_ACTOR_OBS_DIM)
    base[:, 9] = 0.3
    intent = torch.randn(4, DYNAMIC_INTENT_DIM)
    model._mean_from_actor_observation(torch.cat((base, intent), dim=-1)).sum().backward()
    gradient = model.response_adapter[0].weight.grad
    assert gradient is not None
    start = RESPONSE_CONTEXT_DIM
    assert torch.count_nonzero(gradient[:, :start]) == 0
    assert torch.count_nonzero(gradient[:, start + LOCOMOTION_INTENT_DIM :]) == 0
    assert torch.count_nonzero(gradient[:, start : start + LOCOMOTION_INTENT_DIM]) > 0
    assert not model.response_adapter[-1].weight.requires_grad


def test_zero_initialized_transition_adapter_preserves_checkpoint_and_is_masked():
    torch.manual_seed(81)
    baseline = _model("future", response_adapter=True)
    transition = _model(
        "future",
        response_adapter=True,
        transition_adapter=True,
    )
    transition.load_state_dict(baseline.state_dict(), strict=True)
    base = torch.randn(3, BASE_ACTOR_OBS_DIM)
    base[:, 9] = 0.3
    intent = torch.zeros(3, DYNAMIC_INTENT_DIM)
    intent[:, -2:] = torch.tensor([0.6, -0.4])
    observation = torch.cat((base, intent), dim=-1)
    torch.testing.assert_close(
        transition._mean_from_actor_observation(observation),
        baseline._mean_from_actor_observation(observation),
        rtol=0.0,
        atol=0.0,
    )
    assert not any(p.requires_grad for p in transition.response_adapter.parameters())
    assert all(p.requires_grad for p in transition.transition_adapter.parameters())

    torch.nn.init.normal_(transition.transition_adapter[-1].weight, std=10.0)
    torch.nn.init.normal_(transition.transition_adapter[-1].bias, std=10.0)
    transition._mean_from_actor_observation(observation)
    residual = transition._last_coordination_residual
    assert float(torch.max(torch.abs(residual)).detach()) <= 0.100001
    torch.testing.assert_close(
        residual[:, [2, 8, 12]],
        torch.zeros_like(residual[:, [2, 8, 12]]),
    )


def test_zero_initialized_adapter_is_exact_base_actor():
    torch.manual_seed(7)
    model = _model("future")
    base = torch.randn(5, BASE_ACTOR_OBS_DIM)
    intent = torch.randn(5, DYNAMIC_INTENT_DIM)
    full = torch.cat((base, intent), dim=-1)
    torch.testing.assert_close(
        model._mean_from_actor_observation(full),
        model.actor(base),
        rtol=0.0,
        atol=0.0,
    )


def test_zero_intent_hard_gates_a_nonzero_adapter():
    torch.manual_seed(9)
    model = _model("future")
    torch.nn.init.normal_(model.coordination_adapter[-1].weight)
    torch.nn.init.normal_(model.coordination_adapter[-1].bias)
    base = torch.randn(3, BASE_ACTOR_OBS_DIM)
    full = torch.cat((base, torch.zeros(3, DYNAMIC_INTENT_DIM)), dim=-1)
    torch.testing.assert_close(
        model._mean_from_actor_observation(full),
        model.actor(base),
        rtol=0.0,
        atol=0.0,
    )


def test_zero_locomotion_command_hard_gates_a_nonzero_adapter():
    torch.manual_seed(10)
    model = _model("future")
    torch.nn.init.normal_(model.coordination_adapter[-1].weight)
    torch.nn.init.normal_(model.coordination_adapter[-1].bias)
    base = torch.randn(3, BASE_ACTOR_OBS_DIM)
    base[:, 9:12] = 0.0
    intent = torch.ones(3, DYNAMIC_INTENT_DIM)
    torch.testing.assert_close(
        model._mean_from_actor_observation(torch.cat((base, intent), dim=-1)),
        model.actor(base),
        rtol=0.0,
        atol=0.0,
    )


def test_yaw_recovery_command_does_not_keep_adapter_active_at_stop():
    torch.manual_seed(11)
    model = _model("future")
    torch.nn.init.normal_(model.coordination_adapter[-1].weight)
    torch.nn.init.normal_(model.coordination_adapter[-1].bias)
    base = torch.randn(3, BASE_ACTOR_OBS_DIM)
    base[:, 9:12] = torch.tensor([0.0, 0.0, 0.5])
    intent = torch.ones(3, DYNAMIC_INTENT_DIM)
    torch.testing.assert_close(
        model._mean_from_actor_observation(torch.cat((base, intent), dim=-1)),
        model.actor(base),
        rtol=0.0,
        atol=0.0,
    )


def test_locomotion_gate_scales_residual_below_command_threshold():
    torch.manual_seed(12)
    model = _model("future")
    torch.nn.init.normal_(model.coordination_adapter[-1].weight, std=0.2)
    torch.nn.init.normal_(model.coordination_adapter[-1].bias, std=0.2)
    base = torch.randn(2, BASE_ACTOR_OBS_DIM)
    intent = torch.ones(2, DYNAMIC_INTENT_DIM)
    base[:, 9:12] = torch.tensor([0.10, 0.0, 0.0])
    model._mean_from_actor_observation(torch.cat((base, intent), dim=-1))
    full = model._last_coordination_residual.clone()
    base[:, 9:12] = torch.tensor([0.05, 0.0, 0.0])
    model._mean_from_actor_observation(torch.cat((base, intent), dim=-1))
    torch.testing.assert_close(model._last_coordination_residual, 0.5 * full)


def test_residual_is_bounded_and_anatomically_masked():
    torch.manual_seed(11)
    model = _model("future")
    torch.nn.init.normal_(model.coordination_adapter[-1].weight, std=10.0)
    torch.nn.init.normal_(model.coordination_adapter[-1].bias, std=10.0)
    base = torch.randn(4, BASE_ACTOR_OBS_DIM)
    intent = torch.ones(4, DYNAMIC_INTENT_DIM)
    model._mean_from_actor_observation(torch.cat((base, intent), dim=-1))
    residual = model._last_coordination_residual
    assert residual is not None
    assert float(torch.max(torch.abs(residual)).detach()) <= 0.100001
    torch.testing.assert_close(
        residual[:, [3, 4, 5, 9, 10, 11, 13]],
        torch.zeros_like(residual[:, [3, 4, 5, 9, 10, 11, 13]]),
    )


def test_current_mode_discards_future_half():
    torch.manual_seed(13)
    model = _model("current")
    torch.nn.init.normal_(model.coordination_adapter[-1].weight)
    torch.nn.init.normal_(model.coordination_adapter[-1].bias)
    base = torch.randn(2, BASE_ACTOR_OBS_DIM)
    current = torch.randn(2, UPPER_INTENT_DIM // 2)
    future_a = torch.randn(2, UPPER_INTENT_DIM // 2)
    future_b = torch.randn(2, UPPER_INTENT_DIM // 2)
    locomotion = torch.randn(2, LOCOMOTION_INTENT_DIM)
    out_a = model._mean_from_actor_observation(
        torch.cat((base, current, future_a, locomotion), dim=-1)
    )
    out_b = model._mean_from_actor_observation(
        torch.cat((base, current, future_b, locomotion), dim=-1)
    )
    torch.testing.assert_close(out_a, out_b)


def test_coordination_blend_scales_only_the_residual():
    torch.manual_seed(17)
    model = _model("future")
    torch.nn.init.normal_(model.coordination_adapter[-1].weight, std=0.2)
    torch.nn.init.normal_(model.coordination_adapter[-1].bias, std=0.2)
    base = torch.randn(4, BASE_ACTOR_OBS_DIM)
    intent = 0.03 * torch.randn(4, DYNAMIC_INTENT_DIM)
    observation = torch.cat((base, intent), dim=-1)

    model.coordination_blend = 1.0
    model._mean_from_actor_observation(observation)
    full_residual = model._last_coordination_residual.clone()
    model.coordination_blend = 0.25
    output = model._mean_from_actor_observation(observation)

    torch.testing.assert_close(
        model._last_coordination_residual,
        0.25 * full_residual,
    )
    torch.testing.assert_close(
        output,
        model.actor(base) + 0.25 * full_residual,
    )
    model.coordination_blend = 0.0
    torch.testing.assert_close(
        model._mean_from_actor_observation(observation),
        model.actor(base),
        rtol=0.0,
        atol=0.0,
    )
