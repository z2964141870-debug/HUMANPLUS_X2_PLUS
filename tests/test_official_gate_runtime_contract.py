from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_official_container_exposes_tools_package_to_adapter() -> None:
    """Keep the mounted adapter's sibling package importable in Docker.

    Running ``python tools/official_x2/stage208_official_mujoco_adapter.py``
    places only the leaf script directory on ``sys.path``.  The explicit tools
    root is therefore part of the official-runtime contract, not a host-only
    convenience.
    """

    inner = (
        REPO_ROOT / "tools/official_x2/run_official_gate_inner.sh"
    ).read_text(encoding="utf-8")
    adapter = (
        REPO_ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py"
    ).read_text(encoding="utf-8")

    assert 'export PYTHONPATH="/repo/tools${PYTHONPATH:+:$PYTHONPATH}"' in inner
    assert "from official_x2.skill_handoff_contract import" in adapter


def test_official_gate_uses_init_and_sufficient_timeout() -> None:
    runner = (
        REPO_ROOT / "tools/official_x2/run_official_gate_case.sh"
    ).read_text(encoding="utf-8")
    panel = (
        REPO_ROOT / "tools/official_x2/run_stage340_recovery_u10_gate.sh"
    ).read_text(encoding="utf-8")

    assert 'TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"' in runner
    assert "docker run --rm --init --name" in runner
    assert 'ROS_DOMAIN_BASE="${ROS_DOMAIN_BASE:-190}"' in panel
    assert 'ROS_DOMAIN_ID="$((ROS_DOMAIN_BASE + repeat))"' in panel
    assert "ROS_DOMAIN_BASE + REPEATS must stay within CycloneDDS domain 0..232" in panel
    assert 'HEADING_GAIN="${HEADING_GAIN:-0.0}"' in panel
    assert '"heading_gain": heading_gain' in panel


def test_transition_event_smoke_budget_is_explicit_and_positive() -> None:
    runner = (
        REPO_ROOT / "scripts/run_stage345_transition_event_ab.sh"
    ).read_text(encoding="utf-8")

    assert 'UPDATES="${UPDATES:-1}"' in runner
    assert 'UPDATES must be a positive integer' in runner
    assert '--max_iterations "$UPDATES"' in runner


def test_matched_event_preserves_episode_heading_across_stand_to_move() -> None:
    adapter = (
        REPO_ROOT / "tools/official_x2/stage208_official_mujoco_adapter.py"
    ).read_text(encoding="utf-8")

    assert "if self.args.move_accelerate_seconds <= 0.0 or self.heading_target_rad is None:" in adapter
