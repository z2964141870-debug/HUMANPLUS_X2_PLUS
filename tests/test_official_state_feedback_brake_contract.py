from pathlib import Path


def test_brake_then_policy_is_velocity_feedback_then_main_zero_hold():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/official_x2/stage208_official_mujoco_adapter.py").read_text(
        encoding="utf-8"
    )
    helper_start = text.index("def _brake_command(")
    helper_end = text.index("def _curriculum_stop_speed(", helper_start)
    helper = text[helper_start:helper_end]
    assert "world_vector_to_body" in helper
    assert "-self.args.stop_brake_gain * body_velocity[:2]" in helper
    assert "self.args.stop_brake_limit" in helper

    branch_start = text.index('elif self.args.stop_controller in (\n                "velocity_brake"')
    branch_end = text.index('elif self.args.stop_controller == "ramp_then_hold":', branch_start)
    branch = text[branch_start:branch_end]
    assert 'self.args.stop_controller == "brake_then_policy"' in branch
    assert "self.stop_hold_latch_s is None" in branch
    assert "measured_speed <= self.args.event_hold_speed" in branch
    assert "double_support or emergency_latch" in branch
    assert "targets, obs, action = self._policy_targets(0.0, 0.0)" in branch
    assert 'policy_slot="stationary"' not in branch


def test_brake_campaign_uses_frozen_historical_single_setting():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_x2_official_state_feedback_brake_failure_cases.sh").read_text(
        encoding="utf-8"
    )
    assert "STOP_CONTROLLER=brake_then_policy" in text
    assert "STOP_BRAKE_GAIN=1.5" in text
    assert "STOP_BRAKE_LIMIT=0.30" in text
    assert "EVENT_HOLD_MIN_SECONDS=0.5" in text
    assert "EVENT_HOLD_SPEED=0.05" in text
    assert "STOP_EMERGENCY_TILT_RAD" not in text
