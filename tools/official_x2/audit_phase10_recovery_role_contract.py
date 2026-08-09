#!/usr/bin/env python3
"""Fail-closed audit for the proposed Phase10 recovery-only A/B.

Phase10 is valid only if the Phase9 ``curriculum_then_policy`` contract calls
the recovery policy after handoff.  Merely loading a different recovery ONNX
is not an intervention when its inference count remains zero.  This tool is
pure/read-only and deliberately does not launch AimDK, MuJoCo, or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_HASHES = {
    "adapter": "d80900b2e7e1fac98aa586dc44c9dfb3d8561592e93167d931b8177afed7e8f9",
    "moving_model": "da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c",
    "stationary_model": "edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565",
    "candidate_recovery_model": "9bc672fc3c535dbe6cd2709cdec4531eb9b61990457172e9c813a8ac713fc0ca",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop_controller_branch(adapter_text: str, controller: str) -> str:
    marker = f'elif self.args.stop_controller == "{controller}":'
    start = adapter_text.find(marker)
    if start < 0:
        # Some related stop modes share one tuple-dispatch branch.
        controller_at = adapter_text.find(f'"{controller}",')
        if controller_at >= 0:
            start = adapter_text.rfind(
                "elif self.args.stop_controller", 0, controller_at
            )
    if start < 0:
        raise ValueError(f"missing stop-controller branch: {controller}")
    end = adapter_text.find("\n            elif self.args.stop_controller", start + len(marker))
    if end < 0:
        end = adapter_text.find("\n            else:", start + len(marker))
    if end < 0:
        raise ValueError(f"cannot bound stop-controller branch: {controller}")
    return adapter_text[start:end]


def _summary_contract(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model",
        "stationary_model",
        "recovery_model",
        "command_vx_mps",
        "prepare_seconds",
        "stand_seconds",
        "move_seconds",
        "move_accelerate_seconds",
        "stop_seconds",
        "stationary_controller",
        "stationary_blend",
        "stop_controller",
        "stop_transition_seconds",
        "stop_intent_decelerate_seconds",
        "action_bias_mode",
        "action_bias",
        "ankle_roll_common_bias",
        "lateral_position_gain",
        "lateral_velocity_gain",
        "heading_gain",
        "pd_profile",
        "pd_kp_multiplier",
        "pd_kd_multiplier",
        "clock_mode",
    )
    return {key: summary.get(key) for key in keys}


def audit(
    *,
    phase9_panel: Path,
    adapter: Path,
    moving_model: Path,
    stationary_model: Path,
    candidate_recovery_model: Path,
) -> dict[str, Any]:
    paths = {
        "adapter": adapter,
        "moving_model": moving_model,
        "stationary_model": stationary_model,
        "candidate_recovery_model": candidate_recovery_model,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hash_match = {
        name: hashes[name] == EXPECTED_HASHES[name] for name in EXPECTED_HASHES
    }
    if not all(hash_match.values()):
        return {
            "status": "blocked_hash_mismatch",
            "hashes": hashes,
            "expected_hashes": EXPECTED_HASHES,
            "hash_match": hash_match,
            "official_runs_started": 0,
        }

    panel = json.loads(phase9_panel.read_text(encoding="utf-8"))
    source = panel.get("groups", {}).get("source", {})
    rows = source.get("rows", [])
    if source.get("valid") != 5 or len(rows) != 5:
        raise ValueError("Phase9 source control is not a complete five-run panel")

    summaries = []
    counts = []
    for row in rows:
        path = Path(row["path"]).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        summaries.append(_summary_contract(summary))
        counts.append(summary.get("policy_slot_inference_counts"))
    contract_identical = all(summary == summaries[0] for summary in summaries[1:])
    recovery_counts = [int(count.get("recovery", -1)) for count in counts]
    stationary_counts = [int(count.get("stationary", -1)) for count in counts]
    main_counts = [int(count.get("main", -1)) for count in counts]

    adapter_text = adapter.read_text(encoding="utf-8")
    curriculum = stop_controller_branch(adapter_text, "curriculum_then_policy")
    brake_blend = stop_controller_branch(adapter_text, "brake_blend_to_policy")
    curriculum_uses_stationary = 'policy_slot="stationary"' in curriculum
    curriculum_uses_recovery = (
        'policy_slot="recovery"' in curriculum
        or "policy_slot=stop_policy_slot" in curriculum
    )
    brake_blend_uses_recovery_router = "policy_slot=stop_policy_slot" in brake_blend

    proposed_intervention_has_authority = bool(
        contract_identical
        and all(value > 0 for value in recovery_counts)
        and curriculum_uses_recovery
    )
    status = (
        "ready"
        if proposed_intervention_has_authority
        else "blocked_noop_recovery_model_under_phase9_contract"
    )
    return {
        "stage": "base_phase10_recovery_role_contract_audit",
        "status": status,
        "hypothesis": (
            "Keeping stationary=source while changing only recovery_model to f005 "
            "isolates post-handoff recovery behavior."
        ),
        "control": {
            "phase9_source_panel": str(phase9_panel.resolve()),
            "valid_runs": source.get("valid"),
            "contract_identical_across_5": contract_identical,
            "contract": summaries[0],
            "policy_slot_counts": counts,
        },
        "hashes": hashes,
        "expected_hashes": EXPECTED_HASHES,
        "hash_match": hash_match,
        "authority_audit": {
            "main_counts": main_counts,
            "stationary_counts": stationary_counts,
            "recovery_counts": recovery_counts,
            "curriculum_then_policy_uses_stationary": curriculum_uses_stationary,
            "curriculum_then_policy_uses_recovery": curriculum_uses_recovery,
            "brake_blend_to_policy_uses_recovery_router": brake_blend_uses_recovery_router,
            "proposed_intervention_has_authority": proposed_intervention_has_authority,
        },
        "result": (
            "Phase9 source uses recovery 0 times in every episode. Under the exact "
            "curriculum_then_policy branch, post-transition inference is hard-coded "
            "to stationary, so substituting RECOVERY_MODEL is a strict no-op."
        ),
        "alternative_rejected": (
            "brake_blend_to_policy routes through recovery_model, but switching to it "
            "changes the frozen Phase9 stop contract and is not the requested single-variable A/B."
        ),
        "official_runs_started": 0,
        "training_updates": 0,
        "next": (
            "Require a separately preregistered controller-role change or a new contract "
            "whose source control already exercises recovery; do not run five no-op episodes."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase9-panel", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--moving-model", type=Path, required=True)
    parser.add_argument("--stationary-model", type=Path, required=True)
    parser.add_argument("--candidate-recovery-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        phase9_panel=args.phase9_panel,
        adapter=args.adapter,
        moving_model=args.moving_model,
        stationary_model=args.stationary_model,
        candidate_recovery_model=args.candidate_recovery_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
