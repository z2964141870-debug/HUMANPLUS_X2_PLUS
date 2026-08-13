import hashlib
import json
from pathlib import Path


def test_stop_kernel_manifest_is_exact_and_scoped_to_stop_only():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "stage264_stop_kernel_candidate/manifest.json").read_text())
    assert manifest["status"] == "LOCAL_STOP_KERNEL_CANDIDATE_ONLY"
    assert manifest["validation"]["stop_gate_passes"] == 24
    assert manifest["validation"]["full_gate_passes"] == 20
    assert manifest["controller"]["stationary_actor_used"] is False
    assert manifest["permissions"]["default_adapter_change_allowed"] is False
    assert manifest["permissions"]["training_unlocked"] is False
    assert manifest["permissions"]["hardware_deployment_unlocked"] is False
    assert manifest["remote_operations"] == {
        "baidu_access_or_upload": False,
        "github_remote_access_or_push": False,
    }
    for record in manifest["artifacts"].values():
        path = Path(record["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_stop_kernel_runtime_contract_is_complete():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "stage264_stop_kernel_candidate/manifest.json").read_text())
    assert manifest["runtime_environment"] == {
        "STOP_CONTROLLER": "brake_then_policy",
        "STOP_BRAKE_GAIN": "1.5",
        "STOP_BRAKE_LIMIT": "0.30",
        "STOP_BRAKE_TEMPLATE_SPEED": "0.30",
        "STOP_BRAKE_TEMPLATE_FLOOR": "0.25",
        "EVENT_HOLD_MIN_SECONDS": "0.5",
        "EVENT_HOLD_SPEED": "0.05",
    }
