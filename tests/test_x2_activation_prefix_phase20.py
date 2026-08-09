from __future__ import annotations

import json
from pathlib import Path


def test_phase20_report_attributes_failure_to_port_not_protocol():
    path = Path(__file__).resolve().parents[1] / "reports/retarget/x2_native_activation_prefix_phase20.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["capture_attempt"]["subscription_ready"] is True
    assert report["capture_attempt"]["joint_controller_ack"] is True
    assert report["capture_attempt"]["rl_controller_ack"] is False
    assert report["environment_failure"]["error"] == "0.0.0.0:51822 is already in use"
    assert report["truth_boundary"]["failure_does_not_test_activation_order"] is True


def test_phase20_stops_without_touching_foreign_container_or_physics():
    path = Path(__file__).resolve().parents[1] / "reports/retarget/x2_native_activation_prefix_phase20.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["environment_failure"]["read_only_docker_observation"]["action_taken"].startswith("none")
    assert report["truth_boundary"]["foreign_container_not_killed"] is True
    assert report["truth_boundary"]["no_retry"] is True
    assert report["truth_boundary"]["no_prescribed_or_free_physics"] is True
    assert report["decision"]["status"] == "PHASE20_ENVIRONMENT_PORT_COLLISION_NO_PHYSICS"
