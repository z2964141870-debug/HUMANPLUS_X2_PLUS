import json
from pathlib import Path

import onnxruntime as ort

from official_x2.run_phase28_stage250_extended_direct import (
    MOVE_TICKS,
    STAND_TICKS,
    STOP_TICKS,
    TOTAL_TICKS,
    historical_action_preflight,
    stage_at_tick,
)


ROOT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
TRACE = ROOT / "results/official_native_strict_20260807/stage250_video_straight.json"
ACTOR = ROOT / "models/stage219_s2600_actor.onnx"
STATIONARY = ROOT / "models/stand_backend_scratch_i150_actor.onnx"
TEMPLATE = Path(
    "/home/humanplus/x2_teleop_final/x2_sonic/data/processed/"
    "x2_official_forward_gait_phase_template_15dof.npz"
)


def test_stage_schedule_is_frozen_stage250_contract():
    assert (STAND_TICKS, MOVE_TICKS, STOP_TICKS, TOTAL_TICKS) == (100, 200, 400, 700)
    assert stage_at_tick(0) == ("stand", 0.0, 0.0)
    assert stage_at_tick(99)[0] == "stand"
    assert stage_at_tick(100) == ("move", 0.0, 0.3)
    assert stage_at_tick(299)[0] == "move"
    assert stage_at_tick(300) == ("stop", 0.0, 0.0)
    assert stage_at_tick(699)[0] == "stop"


def test_historical_stage250_action_contract_replays_exactly():
    payload = json.loads(TRACE.read_text(encoding="utf-8"))
    main = ort.InferenceSession(str(ACTOR), providers=["CPUExecutionProvider"])
    stationary = ort.InferenceSession(str(STATIONARY), providers=["CPUExecutionProvider"])
    result = historical_action_preflight(payload, main, stationary, TEMPLATE)
    assert result["decoded_rows"] == 700
    assert result["consistent_within_float32_5e7"]
    assert result["first_clip_abs_max"] <= 1.0
