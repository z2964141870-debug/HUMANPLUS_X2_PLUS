#!/usr/bin/env python3
"""Compare two simulator traces and emit a machine-readable exactness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.atol < 0.0:
        raise ValueError("--atol must be non-negative")
    reference_path = args.reference.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path,
        allow_pickle=False,
    ) as candidate:
        reference_keys = set(reference.files)
        candidate_keys = set(candidate.files)
        only_reference = sorted(reference_keys - candidate_keys)
        only_candidate = sorted(candidate_keys - reference_keys)
        comparisons: dict[str, dict[str, object]] = {}
        max_numeric_difference = 0.0
        all_equal = not only_reference and not only_candidate
        for key in sorted(reference_keys & candidate_keys):
            left = reference[key]
            right = candidate[key]
            same_shape = left.shape == right.shape
            numeric = left.dtype.kind in "fiu" and right.dtype.kind in "fiu"
            if not same_shape:
                equal = False
                max_difference = None
            elif numeric:
                max_difference = (
                    float(np.max(np.abs(left - right))) if left.size else 0.0
                )
                max_numeric_difference = max(
                    max_numeric_difference,
                    max_difference,
                )
                equal = max_difference <= args.atol
            else:
                max_difference = None
                equal = bool(np.array_equal(left, right))
            all_equal &= equal
            comparisons[key] = {
                "reference_shape": list(left.shape),
                "candidate_shape": list(right.shape),
                "reference_dtype": str(left.dtype),
                "candidate_dtype": str(right.dtype),
                "max_abs_difference": max_difference,
                "within_tolerance": equal,
            }

    report = {
        "schema_version": 1,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "absolute_tolerance": args.atol,
        "only_in_reference": only_reference,
        "only_in_candidate": only_candidate,
        "common_array_count": len(comparisons),
        "max_numeric_difference": max_numeric_difference,
        "all_arrays_within_tolerance": all_equal,
        "arrays": comparisons,
        "passed": all_equal,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "reference",
                    "candidate",
                    "absolute_tolerance",
                    "common_array_count",
                    "max_numeric_difference",
                    "passed",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not all_equal:
        raise SystemExit("trace equivalence gate failed")


if __name__ == "__main__":
    main()
