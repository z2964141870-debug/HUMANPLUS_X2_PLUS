#!/usr/bin/env python3

import unittest

from summarize_phase_contact_ablation import transition_summary


def row(passed: bool, survival: float) -> dict:
    return {
        "passed": passed,
        "survival_s": survival,
        "fall": None if passed else {"reason": "test"},
        "motion": "synthetic",
    }


class TransitionSummaryTest(unittest.TestCase):
    def test_all_transition_types(self):
        phase = {
            "ff": row(False, 1.0),
            "fp": row(False, 1.0),
            "pf": row(True, 5.0),
            "pp": row(True, 5.0),
        }
        contact = {
            "ff": row(False, 2.0),
            "fp": row(True, 5.0),
            "pf": row(False, 3.0),
            "pp": row(True, 5.0),
        }
        result = transition_summary(phase, contact)
        self.assertEqual(
            result["counts"],
            {"fail_to_fail": 1, "fail_to_pass": 1, "pass_to_fail": 1, "pass_to_pass": 1},
        )
        self.assertEqual(len(result["rescued"]), 1)
        self.assertEqual(len(result["harmed"]), 1)

    def test_mismatched_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "outcome keys differ"):
            transition_summary({"a": row(True, 5.0)}, {"b": row(True, 5.0)})


if __name__ == "__main__":
    unittest.main()
