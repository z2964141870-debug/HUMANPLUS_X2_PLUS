import hashlib

import mujoco
import numpy as np

from official_x2.run_testonly_official_mjcf_suffix_probe import (
    compare_rows,
    integration_state,
    phase_pitch_summary,
    physical_state_sha256,
    restore_integration_state,
)


def _tiny_model_and_data():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.001"/>
          <worldbody>
            <body><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.1, -0.2, 0.7, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:6] = np.arange(6, dtype=np.float64) / 100
    mujoco.mj_forward(model, data)
    return model, data


def test_integration_state_roundtrip_is_bitwise_exact():
    model, source = _tiny_model_and_data()
    state = integration_state(model, source)
    target = mujoco.MjData(model)
    restore_integration_state(model, target, state)
    assert np.array_equal(integration_state(model, target), state)


def test_physical_hash_binds_state_and_asset_manifest():
    _model, data = _tiny_model_and_data()
    state = np.asarray(data.qpos, dtype=np.float64)
    manifest_a = hashlib.sha256(b"assets-a").hexdigest()
    manifest_b = hashlib.sha256(b"assets-b").hexdigest()
    original = physical_state_sha256(state, manifest_a)
    changed = state.copy()
    changed[0] += 1.0e-9
    assert physical_state_sha256(changed, manifest_a) != original
    assert physical_state_sha256(state, manifest_b) != original


def test_row_comparison_is_exact_for_action_event_and_tolerant_for_physics():
    base = {
        "qpos": np.asarray([1.0, 2.0]),
        "qvel": np.asarray([3.0]),
        "root": np.asarray([1.0]),
        "action": np.asarray([0.25], dtype=np.float32),
        "root_pitch_deg": -10.0,
        "controller_event": {"stage": "stop_curriculum", "latch": None},
    }
    same = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in base.items()
    }
    assert compare_rows(base, same)["passed"] is True
    changed = dict(same)
    changed["controller_event"] = {"stage": "stop_stationary", "latch": 2.0}
    result = compare_rows(base, changed)
    assert result["passed"] is False
    assert result["controller_event_exact"] is False


def test_signed_pitch_summary_keeps_start_separate_from_cruise():
    rows = [
        {"stage": "stand", "elapsed_s": 0.4, "root_pitch_deg": -4.0},
        {"stage": "move", "elapsed_s": 0.4, "root_pitch_deg": -7.0},
        {"stage": "move", "elapsed_s": 1.4, "root_pitch_deg": -10.0},
        {"stage": "stop_stationary", "elapsed_s": 2.2, "root_pitch_deg": -6.0},
    ]
    summary = phase_pitch_summary(rows)
    assert summary["stand"]["mean_deg"] == -4.0
    assert summary["start"]["mean_deg"] == -7.0
    assert summary["move"]["mean_deg"] == -10.0
    assert summary["stop"]["mean_deg"] == -6.0
