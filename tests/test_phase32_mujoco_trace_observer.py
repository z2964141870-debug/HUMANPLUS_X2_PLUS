import ctypes

from official_x2.analyze_phase32_closed_prefix import relevant_steps, root_alignment_score
from official_x2.decode_phase32_mujoco_trace import FileHeader, StateRecord


def test_trace_abi_sizes_are_frozen():
    assert ctypes.sizeof(FileHeader) == 256
    assert ctypes.sizeof(StateRecord) == 19024


def test_relevant_steps_start_after_last_reset():
    rows = [
        {"sequence": 0, "call_kind": 1},
        {"sequence": 1, "call_kind": 3},
        {"sequence": 2, "call_kind": 1},
        {"sequence": 3, "call_kind": 2},
        {"sequence": 4, "call_kind": 3},
    ]
    reset, steps = relevant_steps(rows)
    assert reset == 2
    assert [row["sequence"] for row in steps] == [4]


def test_root_alignment_score_is_zero_for_identical_identity_pose():
    physics = {"qpos": [1.0, 2.0, 0.65, 1.0, 0.0, 0.0, 0.0]}
    telemetry = {"root_x_m": 1.0, "root_y_m": 2.0, "root_z_m": 0.65,
                 "root_yaw_rad": 0.0, "root_tilt_rad": 0.0}
    assert root_alignment_score(physics, telemetry) == 0.0
