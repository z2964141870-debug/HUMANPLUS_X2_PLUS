from pathlib import Path


def test_locomotion_zero_stop_keeps_main_actor_and_standing_suffix():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools/official_x2/stage208_official_mujoco_adapter.py").read_text(
        encoding="utf-8"
    )
    start = text.index('elif self.args.stop_controller == "locomotion_zero":')
    end = text.index('elif self.args.stop_controller == "blend_to_policy":', start)
    branch = text[start:end]
    assert 'self._policy_targets(' in branch
    assert '0.0,' in branch
    assert 'policy_slot="main"' in branch
    assert 'force_moving=True' not in branch
    assert 'policy_slot="stationary"' not in branch
    choices = text[text.index('"--stop-controller"'):text.index('help="Controller used after', text.index('"--stop-controller"'))]
    assert '"locomotion_zero"' in choices
