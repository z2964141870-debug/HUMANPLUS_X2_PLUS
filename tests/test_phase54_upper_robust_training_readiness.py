from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools/official_x2/audit_phase54_upper_robust_training_readiness.py"
REPORT = REPO / "reports/retarget/x2_upper_robust_lower_training_readiness_phase54.json"


def _module():
    spec = importlib.util.spec_from_file_location("phase54_audit", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_mapping_and_upper_hook_contract() -> None:
    module = _module()
    assert tuple(module.literal_assignment(module.ADAPTER, "LOWER_JOINTS")) == module.LOWER15
    assert tuple(module.literal_assignment(module.ADAPTER, "ARM_JOINTS")) == module.ARM14
    assert tuple(module.literal_assignment(module.ADAPTER, "HEAD_JOINTS")) == module.HEAD2
    official = tuple(module.literal_assignment(module.ADAPTER, "ISAAC_JOINTS"))
    assert sorted(module.LOWER15 + module.ARM14 + module.HEAD2) == sorted(official)
    hook = module.function_ast(module.HOOK, "patched_apply")
    text = __import__("ast").unparse(hook)
    assert "original_apply(self)" in text
    assert "joint_ids=self._cwi_upper_joint_ids" in text
    assert all(token not in text for token in ("LOWER_JOINTS", "LEG_JOINTS", "WAIST_JOINTS"))


def test_report_is_fail_closed_before_optimizer() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["static_contract_ready"] is True
    assert payload["decision"] == "READY_FOR_DEDICATED_LIVE_ZERO_UPDATE_ONLY"
    assert payload["optimizer_authorized"] is False
    assert payload["mapping_contract"]["env_action_is_lower15"] is True
    assert payload["mapping_contract"]["partition_exact"] is True
    assert len(payload["blockers"]) == 3
    assert payload["training_contract"]["transitions_per_update"] == 1536
    assert payload["training_contract"]["optimizer_steps_per_update"] == 20
