from pathlib import Path
import ast
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/x2_faithful_one_update_phase47.py"


def _module():
    return ast.parse(SOURCE.read_text())


def test_phase47_frozen_workload_is_exactly_one_update():
    text = SOURCE.read_text()
    assert '"envs": 64' in text
    assert '"rollout_steps_per_env": 24' in text
    assert '"transitions": 1536' in text
    assert '"optimizer_minibatch_steps": 20' in text
    assert '"outer_updates": 1' in text


def test_phase47_has_split_and_frozen_parameter_guards():
    text = SOURCE.read_text()
    assert "validate_split(" in text
    assert 'expected_split="train"' in text
    assert "held-out key entered" in text
    assert "frozen dense/reference/FSQ/std" in text
    assert "five_update_authorized\": False" in text
    compile(SOURCE.read_text(), str(SOURCE), "exec")


def test_stage152_trainer_target_is_opt_in():
    launcher = (ROOT / "scripts/run_dcpeft_stage152.sh").read_text()
    assert "FAITHFUL_WBT29_TRAINER_TARGET=" in launcher
    assert '"trainer._target_=${FAITHFUL_WBT29_TRAINER_TARGET}"' in launcher


def test_phase47_launcher_freezes_single_update_and_no_intermediate_save():
    launcher = (ROOT / "scripts/run_x2_faithful_one_update_phase47.sh").read_text()
    for token in ("NUM_ENVS=64", "STEPS_PER_ENV=24", "ITERS=1", "PPO_EPOCHS=5", "NUM_MINI_BATCHES=4"):
        assert token in launcher
    assert "SAVE_FREQUENCY=999999" in launcher
    assert "SAVE_LAST_FREQUENCY=1" in launcher
    assert "source_B0.pt" in launcher


def test_phase47_regression_never_calls_training_or_optimizer_step():
    source = SOURCE.read_text()
    start = source.index("class Phase47BoundedRegressionTrainer")
    bounded = source[start:]
    assert "super().train()" not in bounded
    assert "optimizer.step" not in bounded
    assert '"optimizer_steps": 0' in bounded
    launcher = (ROOT / "scripts/run_x2_faithful_regression_phase47.sh").read_text()
    assert "PRESERVE_LORA_CHECKPOINT=\"${preserve}\"" in launcher
    assert "run_eval pre train" in launcher
    assert "run_eval post held_out" in launcher


def test_phase47_final_report_preserves_stop_and_truth_boundaries():
    report = json.loads(
        (ROOT / "reports/retarget/x2_faithful_exact_s7_one_update_phase47.json").read_text()
    )
    assert report["decision"] == "PASS_ONE_UPDATE_SANITY_STOP_FIVE_UPDATE_LOCKED"
    assert report["intervention"]["optimizer_minibatch_steps"] == 20
    assert report["intervention"]["held_out_optimizer_samples"] == 0
    assert report["numerical_gate"]["pass"] is True
    assert report["dynamic_lunge_gate"]["pass"] is True
    assert report["dynamic_lunge_gate"]["survival_drop_s"] <= 0.1
    assert report["native_gold_regression_gate"]["pass"] is True
    assert report["native_gold_regression_gate"]["worst_survival_drop_fraction"] <= 0.05
    assert report["native_gold_regression_gate"]["worst_tracking_error_increase_fraction"] <= 0.1
    assert report["post_save_reload_gate"]["pass"] is False
    assert report["truth_boundary"]["five_update_authorized"] is False
