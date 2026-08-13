"""Pure helpers for Phase74 single-process paired-lane state preflight."""

from __future__ import annotations

import hashlib

import torch


def paired_lane_ids(seed_index: int, num_pairs: int = 64, *, device: str | torch.device = "cpu"):
    """Return alternating donor/recipient ids for adjacent environment pairs."""

    pair = torch.arange(num_pairs, device=device, dtype=torch.long)
    left = 2 * pair
    right = left + 1
    left_donor = (pair + int(seed_index)) % 2 == 0
    donors = torch.where(left_donor, left, right)
    recipients = torch.where(left_donor, right, left)
    return donors, recipients


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}:{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def pair_diagnostics(
    value: torch.Tensor,
    donors: torch.Tensor,
    recipients: torch.Tensor,
    *,
    atol: float = 0.0,
) -> dict[str, object]:
    """Compute transparent exact/tolerance diagnostics without changing the value."""

    left = value[donors].detach()
    right = value[recipients].detach()
    if left.dtype.is_floating_point:
        left64 = left.to(torch.float64)
        right64 = right.to(torch.float64)
        finite = torch.isfinite(left64) & torch.isfinite(right64)
        matching_nonfinite = (
            (torch.isnan(left64) & torch.isnan(right64))
            | (torch.isposinf(left64) & torch.isposinf(right64))
            | (torch.isneginf(left64) & torch.isneginf(right64))
        )
        nonfinite_mismatch = ~(finite | matching_nonfinite)
        difference = torch.where(finite, (left64 - right64).abs(), torch.zeros_like(left64))
        max_abs = float(difference.max()) if difference.numel() else 0.0
        denominator = float(torch.where(torch.isfinite(left64), left64, torch.zeros_like(left64)).norm())
        relative_l2 = float(difference.norm()) / max(denominator, 1.0e-30)
        per_pair_difference = difference.reshape(difference.shape[0], -1).max(dim=1).values
        per_pair_nonfinite_mismatch = nonfinite_mismatch.reshape(nonfinite_mismatch.shape[0], -1).any(dim=1)
        per_pair = (per_pair_difference > atol) | per_pair_nonfinite_mismatch
        passed = not bool(per_pair.any())
    else:
        mismatch = left != right
        per_pair = mismatch.reshape(mismatch.shape[0], -1).any(dim=1)
        max_abs = float(per_pair.any())
        relative_l2 = max_abs
        passed = not bool(per_pair.any())
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "donor_sha256": tensor_hash(left),
        "recipient_sha256": tensor_hash(right),
        "pair_pass_count": int((~per_pair).sum()),
        "pair_count": int(left.shape[0]),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "threshold": float(atol),
        "passed": passed,
    }


def copy_pair_rows(value: torch.Tensor, donors: torch.Tensor, recipients: torch.Tensor) -> None:
    """Copy batch rows using a clone so overlapping indexing is unambiguous."""

    value[recipients] = value[donors].clone()
