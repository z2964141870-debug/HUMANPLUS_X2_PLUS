import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_panel_phase28.json"
PANEL = REPO / "reports/retarget/x2_wbt_diagnostic_panel.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_phase28_panel_is_complete_and_immutable():
    report = json.loads(REPORT.read_text())
    panel = json.loads(PANEL.read_text())
    assert report["preflight"]["pass"] is True
    assert report["summary"]["completed"] == 24
    assert report["summary"]["blocked"] == 0
    assert {row["id"] for row in report["motions"]} == {row["id"] for row in panel["motions"]}
    assert report["provenance"]["panel"]["sha256"] == sha256(PANEL)
    assert all(row["source_sha256"] == next(p["source_sha256"] for p in panel["motions"] if p["id"] == row["id"]) for row in report["motions"])


def test_phase28_ab_and_tier_counts_are_self_consistent():
    report = json.loads(REPORT.read_text())
    rows = report["motions"]
    assert all(row["old_vs_official"]["exact"] for row in rows)
    assert all(row["old_entry_sha256"] == row["official_entry_sha256"] for row in rows)
    for tier in ("Silver", "Bronze", "Reject"):
        assert report["summary"][tier.lower()] == sum(row["tier_audit"]["tier"] == tier for row in rows)
    assert report["summary"]["silver"] == 0
    assert report["decision"]["wbt_ppo_unlocked"] is False


def test_phase28_did_not_relax_thresholds_or_run_training():
    report = json.loads(REPORT.read_text())
    assert report["offline_metric_contract"]["threshold_revision"] is False
    assert report["offline_metric_contract"]["per_clip_parameter_tuning"] is False
    assert report["truth_boundary"]["physics_steps"] == 0
    assert report["truth_boundary"]["optimizer_steps"] == 0
    assert report["truth_boundary"]["training"] is False
    assert report["truth_boundary"]["real_robot"] is False
