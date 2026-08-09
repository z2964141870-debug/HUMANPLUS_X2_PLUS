import numpy as np

from official_x2.audit_phase35_contact_core import (
    canonical_contact,
    core_and_edge_masks,
    weighted_quantiles,
)


def test_debounce_and_dwell_remove_short_chatter_without_moving_long_edge():
    raw = np.r_[np.zeros(120), np.ones(40), np.zeros(80), np.ones(200), np.zeros(120)].astype(bool)
    result = canonical_contact(raw)
    assert not result[120:160].any()
    assert result[240:440].all()


def test_core_trims_fixed_fifty_ticks_from_both_contact_edges():
    values = np.r_[np.zeros(10), np.ones(200), np.zeros(10)].astype(bool)
    core, edge = core_and_edge_masks(values)
    assert int(np.sum(core)) == 100
    assert int(np.sum(edge)) == 100
    assert core[60:160].all()
    assert not np.any(core & edge)


def test_short_contact_window_has_no_load_bearing_core():
    values = np.r_[np.zeros(10), np.ones(99), np.zeros(10)].astype(bool)
    core, edge = core_and_edge_masks(values)
    assert not core.any()
    assert int(np.sum(edge)) == 99


def test_weighted_quantile_downweights_fast_low_force_contact_point():
    result = weighted_quantiles([0.02, 0.04, 1.0], [100.0, 100.0, 1.0])
    assert result["mean"] < 0.04
    assert result["p95"] < 0.05
