import json
from pathlib import Path

import pytest

from official_x2.run_phase30_mujoco_version_probe import (
    EXPECTED_HASHES,
    PHYSICS_DT,
    SUBSTEPS,
    TICKS,
    version_main_cause,
)


def test_phase30_scope_is_exactly_point_two_seconds():
    assert TICKS == 10
    assert SUBSTEPS == 20
    assert TICKS * SUBSTEPS * PHYSICS_DT == pytest.approx(0.2)


def test_version_cause_requires_all_three_half_reductions():
    assert version_main_cause([0.5, 0.6, 0.7])
    assert not version_main_cause([0.49, 0.9, 0.9])
    with pytest.raises(ValueError):
        version_main_cause([0.5, 0.5])


def test_frozen_phase28_asset_hashes_are_complete():
    assert set(EXPECTED_HASHES) == {
        "scene", "historical", "phase28_control", "phase28_substeps", "vendor_mujoco337"
    }
    assert all(len(value) == 64 for value in EXPECTED_HASHES.values())


def test_frozen_result_excludes_version_as_main_cause():
    path = Path(
        "/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/reports/official_x2/"
        "phase30_mujoco_version_probe.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["runtime"]["exact_vendor_library_preloaded"]
    assert result["result"]["decision"] == "VERSION_IS_NOT_MAIN_CAUSE"
    assert not result["result"]["all_three_reduced_at_least_50pct"]
    for metric in result["result"]["primary_metrics"].values():
        assert metric["relative_reduction"] == 0.0
