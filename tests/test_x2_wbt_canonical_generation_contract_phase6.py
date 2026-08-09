from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_canonical_generation_contract_phase6 as phase6


def test_source_map_is_static_monotone_and_complete():
    mapping = phase6.canonical_source_map(240)
    assert np.array_equal(mapping[: phase6.STATIC_PREFIX_FRAMES], np.zeros(phase6.STATIC_PREFIX_FRAMES))
    assert np.all(np.diff(mapping) >= 0.0)
    assert mapping[-1] == 239


def test_c2_ramp_matches_registered_terminal_slope_and_curvature():
    coefficient = phase6.c2_ramp_coefficients(
        phase6.SOURCE_RAMP_FRAMES,
        phase6.OUTPUT_RAMP_FRAMES,
        phase6.SOURCE_TIME_SCALE,
    )
    a3, a4, a5 = coefficient
    assert np.isclose(a3 + a4 + a5, phase6.SOURCE_RAMP_FRAMES)
    derivative_u = 3 * a3 + 4 * a4 + 5 * a5
    assert np.isclose(derivative_u / phase6.OUTPUT_RAMP_FRAMES, phase6.SOURCE_TIME_SCALE)
    assert np.isclose(6 * a3 + 12 * a4 + 20 * a5, 0.0)


def test_paired_gate_enforces_survival_and_slip_stop_rules():
    baseline = {"simulated_duration_s": 2.0, "stance_slip_p95_mps": 0.05}
    assert phase6.paired_gate(
        baseline, {"simulated_duration_s": 1.901, "stance_slip_p95_mps": 0.070}
    )["pass"]
    assert not phase6.paired_gate(
        baseline, {"simulated_duration_s": 1.899, "stance_slip_p95_mps": 0.070}
    )["pass"]
    assert not phase6.paired_gate(
        baseline, {"simulated_duration_s": 2.0, "stance_slip_p95_mps": 0.071}
    )["pass"]


def test_checked_report_stops_before_ground_and_training():
    report_path = REPO / "reports/retarget/x2_wbt_canonical_generation_contract_phase6.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["truth_boundary"]["contact_is_model_estimate_not_hardware_grf"] is True
    assert report["canonical_model_gate"]["pass"] is True
    permutation = report["canonical_model_gate"]["official_rl_29_to_mjcf_31_indices"]
    assert len(permutation) == 29 and len(set(permutation)) == 29
    assert report["phase5_ground_prerequisite"]["ground_offline_all_actions"] is False
    assert report["decision"]["checks"]["root_ground_contact_stage_executed"] is False
    assert report["decision"]["checks"]["training_or_teacher_executed"] is False
    assert report["decision"]["promote_generation_contract"] is False
