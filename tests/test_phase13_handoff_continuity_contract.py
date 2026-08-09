import hashlib
from pathlib import Path
import json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_handoff_blend_is_default_off_and_only_in_curriculum_branch() -> None:
    adapter = (
        REPO_ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py"
    ).read_text(encoding="utf-8")
    parser = adapter[adapter.index("def parse_args()") : adapter.index("def main()")]
    assert '"--curriculum-recovery-handoff-blend-seconds"' in parser
    assert "default=0.0" in parser

    start = adapter.index('elif self.args.stop_controller == "curriculum_then_policy":')
    end = adapter.index('elif self.args.stop_controller == "ramp_policy":', start)
    branch = adapter[start:end]
    assert adapter.count("blend_curriculum_recovery_targets(") == 1
    assert "self.stop_hold_targets = dict(targets)" in branch
    assert "targets = blend_curriculum_recovery_targets(" in branch
    assert "action = np.asarray(" in branch
    assert "normalized_action_from_physical_targets(" in branch
    assert "self.issued_actions[stop_slot] = action.copy()" in branch
    assert "self.previous_actions[stop_slot] = action.copy()" in branch
    assert "targets, obs, action = self._policy_targets(" in branch
    # Blending happens after policy inference, then explicitly synchronizes
    # both action-history buffers with the target that was actually executed.
    assert branch.index("targets, obs, action = self._policy_targets(", branch.index("else:")) < branch.index(
        "targets = blend_curriculum_recovery_targets("
    )


def test_official_runtime_passes_handoff_blend_as_an_explicit_parameter() -> None:
    outer = (REPO_ROOT / "tools/official_x2/run_official_gate_case.sh").read_text(
        encoding="utf-8"
    )
    inner = (REPO_ROOT / "tools/official_x2/run_official_gate_inner.sh").read_text(
        encoding="utf-8"
    )
    assert "CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS" in outer
    assert 'CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS="${CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS:-0.0}"' in inner
    assert '--curriculum-recovery-handoff-blend-seconds "$CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS"' in inner


def test_stop_trace_records_physical_target_separately_from_actor_action() -> None:
    adapter = (
        REPO_ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py"
    ).read_text(encoding="utf-8")
    assert '"physical_lower_target_rad"' in adapter
    assert '"unblended_policy_action"' in adapter
    assert 'physical_targets=targets' in adapter
    assert '"action": [] if action is None else action.tolist()' in adapter


def test_phase13_runner_freezes_one_variable_source_recovery_ab() -> None:
    runner = (
        REPO_ROOT / "tools/official_x2/run_phase13_handoff_continuity_ab.sh"
    ).read_text(encoding="utf-8")
    contract = REPO_ROOT / "tools/official_x2/skill_handoff_contract.py"
    # Historical Phase13 must retain the adapter digest it actually executed;
    # later default-off evidence hooks may legitimately change the live file.
    assert 'ADAPTER_SHA="a929ccec61f91af9d8492b11c4e50c7a61379d27a69b02327b984b2a9dd6759f"' in runner
    assert f'HANDOFF_SHA="{_sha256(contract)}"' in runner
    assert "REPEATS=5" in runner
    assert "labels=(blend0p0 blend0p5)" in runner
    assert "blend_seconds=(0.0 0.5)" in runner
    assert "STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx" in runner
    assert "RECOVERY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx" in runner
    assert 'CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS="$blend"' in runner
    assert '"single_variable":"curriculum_recovery_handoff_blend_seconds 0.0 vs 0.5"' in runner
    assert '"training_unlocked":False' in runner
    assert "train_x2" not in runner


def test_phase13_official_result_proves_mechanism_but_not_physical_gate() -> None:
    result = json.loads(
        (
            REPO_ROOT / "reports/official_x2/phase13_handoff_continuity_ab.json"
        ).read_text(encoding="utf-8")
    )
    assert result["decision"]["all_10_valid"]
    assert result["decision"]["all_slot_counts_exact"]
    assert result["decision"]["candidate_physical_first_tick_continuous"]
    assert result["decision"]["candidate_history_synchronized"]
    assert not result["decision"]["candidate_improves_full_gate"]
    control = result["groups"]["blend0p0"]["aggregate"]
    candidate = result["groups"]["blend0p5"]["aggregate"]
    assert control["metrics"]["physical_target_handoff_delta_l2_rad"]["median"] > 0.4
    assert candidate["metrics"]["physical_target_handoff_delta_l2_rad"]["max"] == 0.0
    assert candidate["metrics"]["physical_target_max_step_l2_rad_through_0p5s"]["max"] < 0.04
    assert candidate["metrics"]["next_previous_action_sync_linf_max"]["max"] == 0.0
    assert control["full"] == candidate["full"] == 0
    assert control["subgates"]["stop"] == candidate["subgates"]["stop"] == 0
