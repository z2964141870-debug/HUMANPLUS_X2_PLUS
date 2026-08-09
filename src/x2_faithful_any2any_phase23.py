"""Faithful WBT29 and native-Gold ingestion contracts for X2.

This module is deliberately independent of Isaac Lab.  It provides the
name-driven boundary that a live environment can call, plus a split-locked
hook for an already-created SONIC MotionLib.  It never advances simulation,
creates an optimizer, or mutates the source Gold files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import torch

from x2_native_gold_motionlib_adapter import apply_recorded_state_adapter


POLICY_TERM_ORDER = ("gravity_dir", "base_ang_vel", "joint_pos", "joint_vel", "actions")
POLICY_TERM_DIMS = {
    "gravity_dir": 3,
    "base_ang_vel": 3,
    "joint_pos": 29,
    "joint_vel": 29,
    "actions": 29,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(names: Sequence[str], expected: int, label: str) -> tuple[str, ...]:
    result = tuple(names)
    if len(result) != expected or len(set(result)) != expected:
        raise ValueError(f"{label} must contain {expected} unique names")
    return result


@dataclass(frozen=True)
class WBT29PolicyContract:
    """Name-driven 31/29/source permutation and head-exclusion contract."""

    official31: tuple[str, ...]
    target29: tuple[str, ...]
    source29: tuple[str, ...]
    head2: tuple[str, ...]

    @classmethod
    def build(
        cls,
        official31: Sequence[str],
        target29: Sequence[str],
        source29: Sequence[str],
        head2: Sequence[str],
    ) -> "WBT29PolicyContract":
        contract = cls(
            _unique(official31, 31, "official31"),
            _unique(target29, 29, "target29"),
            _unique(source29, 29, "source29"),
            _unique(head2, 2, "head2"),
        )
        if set(contract.target29) != set(contract.source29):
            raise ValueError("target29 and source29 must contain the same semantic joints")
        if set(contract.official31) != set(contract.target29) | set(contract.head2):
            raise ValueError("official31 must partition exactly into target29 and head2")
        return contract

    @staticmethod
    def _indices(source_names: Sequence[str], destination_names: Sequence[str]) -> tuple[int, ...]:
        return tuple(source_names.index(name) for name in destination_names)

    @property
    def official_to_source_indices(self) -> tuple[int, ...]:
        return self._indices(self.official31, self.source29)

    @property
    def official_to_target_indices(self) -> tuple[int, ...]:
        return self._indices(self.official31, self.target29)

    @property
    def target_to_source_indices(self) -> tuple[int, ...]:
        return self._indices(self.target29, self.source29)

    @property
    def source_to_target_indices(self) -> tuple[int, ...]:
        return self._indices(self.source29, self.target29)

    @property
    def head_indices(self) -> tuple[int, ...]:
        return self._indices(self.official31, self.head2)

    @staticmethod
    def _select(values: torch.Tensor, indices: Sequence[int], expected: int) -> torch.Tensor:
        if values.shape[-1] != expected:
            raise ValueError(f"expected last dimension {expected}, got {values.shape[-1]}")
        index = torch.as_tensor(indices, device=values.device, dtype=torch.long)
        return values.index_select(-1, index)

    def official_to_source(self, values31: torch.Tensor) -> torch.Tensor:
        return self._select(values31, self.official_to_source_indices, 31)

    def target_to_source(self, values29: torch.Tensor) -> torch.Tensor:
        return self._select(values29, self.target_to_source_indices, 29)

    def source_to_target(self, values29: torch.Tensor) -> torch.Tensor:
        return self._select(values29, self.source_to_target_indices, 29)

    def source_action_to_official(
        self, source_action29: torch.Tensor, sim_nominal31: torch.Tensor
    ) -> torch.Tensor:
        """Scatter a source-semantic policy action while preserving nominal head.

        ``sim_nominal31`` is explicit rather than synthesized so the two head
        coordinates cannot silently become policy outputs.
        """
        if source_action29.shape[-1] != 29 or sim_nominal31.shape[-1] != 31:
            raise ValueError("expected source action 29 and simulator nominal 31")
        prefix = torch.broadcast_shapes(source_action29.shape[:-1], sim_nominal31.shape[:-1])
        source = source_action29.expand(*prefix, 29)
        result = sim_nominal31.expand(*prefix, 31).clone()
        index = torch.as_tensor(
            self.official_to_source_indices, device=result.device, dtype=torch.long
        )
        result.index_copy_(-1, index, source)
        return result

    def adapt_joint_term(self, values: torch.Tensor, semantic_order: str) -> torch.Tensor:
        """Adapt current values or any leading-dimension history into source29."""
        if semantic_order == "official31":
            return self.official_to_source(values)
        if semantic_order == "target29":
            return self.target_to_source(values)
        if semantic_order == "source29":
            if values.shape[-1] != 29:
                raise ValueError("source29 term does not have 29 values")
            return values
        raise ValueError(f"unknown semantic order: {semantic_order}")

    def adapt_policy_history(
        self,
        terms: Mapping[str, torch.Tensor],
        *,
        joint_order: str = "official31",
        action_order: str = "source29",
    ) -> dict[str, torch.Tensor]:
        if tuple(terms) != POLICY_TERM_ORDER:
            raise ValueError(f"policy terms must be ordered exactly as {POLICY_TERM_ORDER}")
        adapted = dict(terms)
        adapted["joint_pos"] = self.adapt_joint_term(terms["joint_pos"], joint_order)
        adapted["joint_vel"] = self.adapt_joint_term(terms["joint_vel"], joint_order)
        adapted["actions"] = self.adapt_joint_term(terms["actions"], action_order)
        history_shapes = {value.shape[:-1] for value in adapted.values()}
        if len(history_shapes) != 1:
            raise ValueError(f"policy term history prefixes differ: {history_shapes}")
        for name, expected in POLICY_TERM_DIMS.items():
            if adapted[name].shape[-1] != expected:
                raise ValueError(f"{name} must have dimension {expected}")
        return adapted

    def flatten_policy_history(self, adapted: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Flatten each term's final history and feature axes, then concatenate."""
        if tuple(adapted) != POLICY_TERM_ORDER:
            raise ValueError(f"policy terms must be ordered exactly as {POLICY_TERM_ORDER}")
        histories = {value.shape[-2] for value in adapted.values()}
        if len(histories) != 1:
            raise ValueError("all policy terms must use the same history length")
        return torch.cat(
            [adapted[name].flatten(start_dim=-2) for name in POLICY_TERM_ORDER], dim=-1
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "official31_order": list(self.official31),
            "policy_source29_order": list(self.source29),
            "sim_target29_order": list(self.target29),
            "head_excluded2": list(self.head2),
            "official_to_source_indices": list(self.official_to_source_indices),
            "source_to_target_indices": list(self.source_to_target_indices),
            "head_indices": list(self.head_indices),
            "policy_term_order": list(POLICY_TERM_ORDER),
            "policy_term_dims": POLICY_TERM_DIMS,
            "history_length": 10,
            "flat_policy_dim": 930,
            "policy_action_dim": 29,
            "sim_action_dim": 31,
            "head_rule": "absent from observation/action/history; simulator coordinates copied from explicit nominal31",
        }


@dataclass(frozen=True)
class GoldSplitSpec:
    name: str
    source_pkl: Path
    sha256: str
    key_prefix: str
    optimizer_eligible: bool


class ImmutableGoldMotionLibHook:
    """Split-locked state hook for a native SONIC MotionLib instance."""

    def __init__(self, spec: GoldSplitSpec):
        if spec.name not in ("train", "held_out"):
            raise ValueError("Gold split must be train or held_out")
        self.spec = spec

    def validate_source(self) -> dict[str, Any]:
        if _sha256(self.spec.source_pkl) != self.spec.sha256:
            raise ValueError(f"Gold {self.spec.name} hash differs from frozen contract")
        payload = joblib.load(self.spec.source_pkl)
        keys = list(payload)
        if not keys or not all(key.startswith(self.spec.key_prefix) for key in keys):
            raise ValueError(f"Gold {self.spec.name} contains unexpected sampler keys")
        if not all(entry.get("split") == self.spec.name for entry in payload.values()):
            raise ValueError(f"Gold {self.spec.name} entry labels differ from split")
        return {
            "split": self.spec.name,
            "path": str(self.spec.source_pkl),
            "sha256": self.spec.sha256,
            "sampler_keys": keys,
            "optimizer_eligible": self.spec.optimizer_eligible,
            "frames": int(sum(len(entry["dof"]) for entry in payload.values())),
        }

    def attach(self, motion_lib: Any) -> dict[str, Any]:
        source = self.validate_source()
        if list(motion_lib.curr_motion_keys) != source["sampler_keys"]:
            raise ValueError("live MotionLib keys do not match the split-locked Gold keys")
        result = apply_recorded_state_adapter(motion_lib, self.spec.source_pkl)
        return {**source, "state_adapter": result}
