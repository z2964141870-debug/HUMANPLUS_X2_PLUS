from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "reports/retarget/x2_privileged_posture_stop_teacher_v4_prereg.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_record_matches_current_bytes() -> None:
    payload = json.loads(PREREG.read_text())
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in payload[section].items():
            path = Path(record["path"])
            assert path.is_file(), (section, name, path)
            assert sha256(path) == record["sha256"], (section, name, path)


def test_preregistration_sidecar_and_outputs_are_clean() -> None:
    sidecar = PREREG.with_name(PREREG.name + ".sha256")
    assert sidecar.read_text() == f"{sha256(PREREG)}  {PREREG.name}\n"
    payload = json.loads(PREREG.read_text())
    for name, raw in payload["outputs"].items():
        if name == "overwrite_forbidden":
            continue
        path = Path(raw)
        assert not path.exists(), (name, path)
        assert not path.with_name(path.name + ".sha256").exists(), (name, path)


def test_contract_stays_zero_optimizer_and_never_directly_promotes() -> None:
    payload = json.loads(PREREG.read_text())
    resources = payload["resource_limits"]
    assert resources["optimizer_steps"] == 0
    assert resources["backward_calls"] == 0
    assert resources["checkpoint_writes"] == 0
    permissions = payload["permissions"]
    assert not permissions["ppo_or_long_training_unlocked"]
    assert not permissions["whole_body_training_unlocked"]
    assert not permissions["official_promotion_unlocked"]
    assert not permissions["deployment_unlocked"]
