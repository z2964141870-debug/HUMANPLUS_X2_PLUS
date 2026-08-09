from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from official_x2.audit_official_x2_event_replay_phase14 import (
    DEFAULT_CONTROLLER_LOG,
    DEFAULT_MANIFEST,
    DEFAULT_SOURCE,
    audit_capture,
    quantiles,
    render,
)


def test_quantiles_use_requested_scale():
    result = quantiles(np.asarray([1, 2, 3], dtype=np.float64), 0.5)
    assert result["min"] == 0.5
    assert result["p50"] == 1.0
    assert result["max"] == 1.5


def test_phase14_final_capture_passes_active_wbt29_contract():
    report = audit_capture(DEFAULT_SOURCE, DEFAULT_MANIFEST, DEFAULT_CONTROLLER_LOG)
    assert report["decision"]["status"] == "PHASE14_WBT29_EVENT_CAPTURE_PASSED_WITH_RECORDER_SOURCE_DRIFT_BOUNDARY"
    assert report["snapshots"]["count"] == 999
    assert report["events"]["count"] == 40083
    assert report["events"]["by_group"]["leg"]["events"] == 10022
    assert report["events"]["by_group"]["waist"]["events"] == 10021
    assert report["events"]["by_group"]["arm"]["events"] == 10020
    assert report["events"]["by_group"]["head"]["events"] == 10020
    assert report["gate"]["passed"] is True
    assert report["gate"]["checks"]["global_event_index_exact"] is True
    assert report["gate"]["checks"]["effective_wbt29_after_warmup"] is True
    assert report["gate"]["checks"]["capture_runtime_artifacts_recoverable_exactly"] is True
    assert report["gate"]["checks"]["current_recorder_source_matches_capture_manifest"] is False
    assert report["recorder_provenance_boundary"]["capture_time_recorder_source_archived_separately"] is False
    assert report["recorder_provenance_boundary"]["exact_capture_recorder_source_reproducible"] is False
    assert report["downstream_dependency_boundary"]["phase15_requires_current_recorder_source_hash"] is False


def test_phase14_truth_boundaries_are_explicit():
    report = audit_capture(DEFAULT_SOURCE, DEFAULT_MANIFEST, DEFAULT_CONTROLLER_LOG)
    assert report["header_boundary"]["schema_header_available_ratio"] == 1.0
    assert report["header_boundary"]["header_populated_ratio"] == 0.0
    assert report["header_boundary"]["all_stamp_and_sequence_fields_zero"] is True
    assert report["gate"]["checks"]["publisher_header_populated"] is False
    assert report["gate"]["checks"]["active_31_joint_control_available"] is False
    assert report["representation_boundary"]["effective_valid_after_two_warmup_events"] == 29
    assert report["events"]["by_group"]["head"]["changed_joint_counts"] == {"0": 10020}
    markdown = render(report)
    assert "不能声称获得了31个主动关节" in markdown
    assert "source rollout 自身稳定不能冒充 replay 稳定" in markdown
    assert "不能声称 recorder 源码可精确重现" in markdown


def test_saved_report_is_consistent_when_present():
    path = Path(__file__).resolve().parents[1] / "reports/retarget/x2_native_event_replay_phase14.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gate"]["passed"] is True
    assert saved["recorder_provenance_boundary"]["current_source_matches_capture"] is False
    assert saved["inputs"]["source_sha256"] == "2b72732acfe8ce427a976ec5f9b7d2366cca9c3835a70498ba28f58d814a5681"
