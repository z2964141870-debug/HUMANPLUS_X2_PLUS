from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from dcpeft_reward_contract import load_contract, materialize_bad_contract


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC = ROOT / "configs" / "reward_contract_dual_v1.json"
BAD = ROOT / "configs" / "reward_contract_bad_materialized_v1.json"


class RewardContractTest(unittest.TestCase):
    def test_unique_and_complete_against_stage152_config_terms(self):
        import yaml

        config = yaml.safe_load(
            Path(
                "/home/humanplus/x2_teleop_final/x2_sonic/logs/ppo_dryrun/"
                "x2_stage152_B_pilot_seed0_v1/config.yaml"
            ).read_text()
        )
        active_terms = tuple(
            name
            for name, value in config["manager_env"]["rewards"].items()
            if name != "_target_" and isinstance(value, dict) and "weight" in value
        )
        contract = load_contract(SEMANTIC)
        contract.validate_terms(active_terms)

    def test_scalar_vector_equivalence_synthetic(self):
        contract = load_contract(SEMANTIC)
        names = tuple(contract.term_to_group)
        generator = torch.Generator().manual_seed(7)
        # IsaacLab integrates each reward contribution over control dt (0.02 s).
        # Use that real scale so the test exercises the actual float32 budget.
        per_term = torch.randn(32, len(names), generator=generator) * 0.02
        index = {name: i for i, name in enumerate(names)}
        grouped = torch.stack(
            [
                per_term[:, [index[name] for name in terms]].sum(dim=1)
                for terms in contract.group_terms
            ],
            dim=1,
        )
        max_abs_error = (grouped.sum(dim=1) - per_term.sum(dim=1)).abs().max()
        self.assertLessEqual(float(max_abs_error), 1e-6)

    def test_bad_grouping_is_deterministic_and_preserves_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.json"
            two = Path(directory) / "two.json"
            materialize_bad_contract(SEMANTIC, one)
            materialize_bad_contract(SEMANTIC, two)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            semantic = load_contract(SEMANTIC)
            bad = load_contract(one)
            self.assertEqual(set(semantic.term_to_group), set(bad.term_to_group))
            self.assertNotEqual(semantic.group_terms, bad.group_terms)


if __name__ == "__main__":
    unittest.main()
