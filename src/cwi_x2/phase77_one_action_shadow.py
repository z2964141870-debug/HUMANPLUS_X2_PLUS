"""Pure helpers for the Phase77 one-action paired-lane technical shadow."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from cwi_x2.phase76_pairing_preflight import copy_pair_rows, pair_diagnostics


FORBIDDEN_POST_PREFIXES = (
    "dynamic.reward.",
    "dynamic.reward_term.",
    "dynamic.command.metrics.",
)
FORBIDDEN_POST_EXACT = {"dynamic.env.reward_buf"}


def pair_shared_action(
    source_action: torch.Tensor,
    donors: torch.Tensor,
    recipients: torch.Tensor,
) -> torch.Tensor:
    """Return a finite 15-D action with each recipient copied from its donor."""

    if source_action.ndim != 2 or source_action.shape != (128, 15):
        raise ValueError(f"Phase77 expected source action [128,15], got {tuple(source_action.shape)}")
    if not torch.isfinite(source_action).all():
        raise ValueError("Phase77 source action must be finite")
    shared = torch.clamp(source_action.detach().clone(), -1.0, 1.0)
    copy_pair_rows(shared, donors, recipients)
    if not pair_diagnostics(shared, donors, recipients, atol=0.0)["passed"]:
        raise RuntimeError("Phase77 pair-shared action is not bit-exact")
    return shared


def technical_post_tensors(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove all reward/performance values before post-step pair diagnostics."""

    filtered = {
        name: value
        for name, value in values.items()
        if name not in FORBIDDEN_POST_EXACT
        and not any(name.startswith(prefix) for prefix in FORBIDDEN_POST_PREFIXES)
    }
    if any(name in filtered for name in FORBIDDEN_POST_EXACT):
        raise RuntimeError("Phase77 retained forbidden reward buffer")
    if any(any(name.startswith(prefix) for prefix in FORBIDDEN_POST_PREFIXES) for name in filtered):
        raise RuntimeError("Phase77 retained a forbidden reward/performance field")
    return filtered


def reward_state_unchanged(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> bool:
    """Compare reward buffers in memory without serializing values or hashes."""

    return set(before) == set(after) and all(torch.equal(before[name], after[name]) for name in before)


def native_bool(value: object) -> bool:
    """Normalize tensor booleans before JSON serialization."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Phase77 expected a scalar boolean tensor")
        return bool(value.item())
    return bool(value)
