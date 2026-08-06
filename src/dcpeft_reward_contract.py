"""Reward decomposition contract used by the isolated DC-PEFT experiments.

The old X2/SONIC tree remains read-only. This module converts IsaacLab's
per-term reward bookkeeping into a vector while proving that its sum equals
the scalar reward returned by the unmodified environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RewardContract:
    contract_id: str
    group_names: tuple[str, ...]
    group_terms: tuple[tuple[str, ...], ...]
    tolerance: float

    @property
    def term_to_group(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group_index, terms in enumerate(self.group_terms):
            for term in terms:
                if term in result:
                    raise ValueError(f"reward term assigned more than once: {term}")
                result[term] = group_index
        return result

    def validate_terms(self, active_terms: Iterable[str]) -> None:
        active = tuple(active_terms)
        active_set = set(active)
        assigned = self.term_to_group
        missing = sorted(active_set - assigned.keys())
        extra = sorted(assigned.keys() - active_set)
        if missing or extra:
            raise ValueError(
                "reward contract does not exactly cover active terms: "
                f"missing={missing}, extra={extra}"
            )
        if len(active) != len(active_set):
            raise ValueError("environment contains duplicate active reward term names")


def load_contract(path: str | Path) -> RewardContract:
    path = Path(path)
    payload = json.loads(path.read_text())
    groups = payload.get("groups")
    if not isinstance(groups, Mapping) or len(groups) < 2:
        raise ValueError("reward contract needs at least two non-empty named groups")
    group_names = tuple(str(name) for name in groups)
    group_terms = tuple(tuple(str(term) for term in groups[name]) for name in group_names)
    if any(not terms for terms in group_terms):
        raise ValueError(
            f"reward contract contains an empty group; materialize generated controls first: {path}"
        )
    tolerance = float(payload.get("notes", {}).get("runtime_equivalence_tolerance", 1e-6))
    contract = RewardContract(
        contract_id=str(payload.get("contract_id", path.stem)),
        group_names=group_names,
        group_terms=group_terms,
        tolerance=tolerance,
    )
    # Detect duplicates even before environment terms are known.
    _ = contract.term_to_group
    return contract


def materialize_bad_contract(
    semantic_path: str | Path,
    output_path: str | Path,
    seed: int = 20260728,
) -> dict[str, Any]:
    semantic = json.loads(Path(semantic_path).read_text())
    source_groups = semantic["groups"]
    names = list(source_groups)
    if len(names) != 2:
        raise ValueError("negative-control generator currently expects exactly two groups")
    terms = sorted(term for group in source_groups.values() for term in group)
    random.Random(seed).shuffle(terms)
    first_size = len(source_groups[names[0]])
    payload = {
        "schema_version": 1,
        "contract_id": f"stage152b_bad_grouping_seed_{seed}",
        "description": "Pre-registered deterministic shuffled negative control.",
        "shuffle_seed": seed,
        "groups": {
            "shuffled_a": terms[:first_size],
            "shuffled_b": terms[first_size:],
        },
        "source_contract": str(Path(semantic_path).name),
        "notes": {
            "runtime_equivalence_tolerance": semantic.get("notes", {}).get(
                "runtime_equivalence_tolerance", 1e-6
            )
        },
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload
