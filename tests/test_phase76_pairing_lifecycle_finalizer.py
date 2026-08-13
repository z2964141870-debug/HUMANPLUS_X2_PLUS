from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXEC_LINE = 'exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())'


def replace_last(text: str, old: str, new: str) -> str:
    position = text.rfind(old)
    assert position >= 0
    return text[:position] + new + text[position + len(old) :]


def mechanically_expand_phase76_finalizer() -> str:
    outer_path = ROOT / "tools/retarget/finalize_x2_phase76_pairing_lifecycle_preflight.py"
    outer = outer_path.read_text()
    first: dict[str, object] = {"__file__": str(outer_path)}
    exec(compile(replace_last(outer, EXEC_LINE, "TRANSFORMED_WRAPPER = source"), str(outer_path), "exec"), first)
    wrapper = first["TRANSFORMED_WRAPPER"]
    assert isinstance(wrapper, str)
    second: dict[str, object] = {"__file__": str(outer_path)}
    exec(compile(replace_last(wrapper, EXEC_LINE, "FINAL_SOURCE = source"), str(outer_path), "exec"), second)
    final_source = second["FINAL_SOURCE"]
    assert isinstance(final_source, str)
    return final_source


def test_phase76_finalizer_adds_all_lifecycle_gates() -> None:
    text = (ROOT / "tools/retarget/finalize_x2_phase76_pairing_lifecycle_preflight.py").read_text()
    assert "EXPECTED_PHASE75_SHA256" in text
    for token in (
        "raw_returncode",
        "autonomous_exit",
        "timed_out",
        "term_sent",
        "kill_sent",
        "forced_cleanup",
    ):
        assert token in text


def test_phase76_finalizer_only_unlocks_phase77_shadow_preregistration() -> None:
    frozen = (ROOT / "tools/retarget/finalize_x2_phase75_pairing_preflight.py").read_text()
    assert 'source.replace("phase75_shadow_preregistration_unlocked", "phase76_shadow_preregistration_unlocked")' in frozen
    text = (ROOT / "tools/retarget/finalize_x2_phase76_pairing_lifecycle_preflight.py").read_text()
    assert 'source.replace("Phase75", "Phase76").replace("phase75", "phase76")' in text
    assert 'f\'"phase77_{permission_suffix}"\'' in text


def test_mechanically_expanded_finalizer_enforces_lifecycle_and_permissions() -> None:
    expanded = mechanically_expand_phase76_finalizer()
    for token in (
        'int(row.get("raw_returncode", -1)) == 0',
        'row.get("autonomous_exit") is True',
        'row.get("timed_out") is False',
        'row.get("term_sent") is False',
        'row.get("kill_sent") is False',
        'row.get("forced_cleanup") is False',
        '"phase77_shadow_preregistration_unlocked": valid',
        '"phase77_scientific_preregistration_unlocked": False',
        '"phase77_launch_unlocked": False',
    ):
        assert token in expanded
    compile(expanded, str(ROOT / "tools/retarget/finalize_x2_phase76_pairing_lifecycle_preflight.py"), "exec")


def test_shell_resource_labels_match_expanded_finalizer_exactly() -> None:
    expanded = mechanically_expand_phase76_finalizer()
    assert 'f"phase76_seed{seed}_single_process_pairing_preflight"' in expanded
    shell = (ROOT / "scripts/run_x2_phase76_pairing_lifecycle_preflight.sh").read_text()
    assert '--label "phase76_seed${seed}_single_process_pairing_preflight"' in shell
    assert 'single_process_pairing_lifecycle_preflight"' not in shell
