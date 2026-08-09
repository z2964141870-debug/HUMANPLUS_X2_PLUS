"""Minimal zero-output LoRA for the standard 93D Stage219 actor/critic."""

from __future__ import annotations

import hashlib

import torch
from torch import nn


class ZeroOutputLoRALinear(nn.Module):
    """Frozen dense linear plus a rank-r LoRA residual whose B starts at zero."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0.0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5.0**0.5)
        self.scaling = float(alpha) / float(rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dense = self.base(inputs)
        residual = (inputs @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return dense + self.scaling * residual


def inject_standard93d_lora(model: nn.Module, *, rank: int = 4, alpha: float = 4.0) -> dict:
    """Freeze Stage219 and attach LoRA only to actor/critic dense paths."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scopes: dict[str, list[str]] = {"actor": [], "critic": []}
    for branch_name in ("actor", "critic"):
        branch = getattr(model, branch_name)
        for index in (0, 2, 4, 6):
            layer = branch[index]
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"{branch_name}.{index} is not a dense linear layer")
            branch[index] = ZeroOutputLoRALinear(layer, rank=rank, alpha=alpha)
            scopes[branch_name].append(f"{branch_name}.{index}")
    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Phase57 trainable scope is not LoRA-only")
    return {
        "rank": rank,
        "alpha": alpha,
        "actor_scopes": scopes["actor"],
        "critic_scopes": scopes["critic"],
        "trainable_names": trainable,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def dense_tensor_map(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return canonical dense/std tensors independent of LoRA wrapper nesting."""

    result: dict[str, torch.Tensor] = {"std": model.std.detach()}
    for branch_name in ("actor", "critic"):
        branch = getattr(model, branch_name)
        for index in (0, 2, 4, 6):
            layer = branch[index]
            dense = layer.base if isinstance(layer, ZeroOutputLoRALinear) else layer
            result[f"{branch_name}.{index}.weight"] = dense.weight.detach()
            result[f"{branch_name}.{index}.bias"] = dense.bias.detach()
    return result


def tensor_map_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        array = tensor.cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
