from __future__ import annotations

import json
from pathlib import Path
import tempfile

from dcpeft_upper_panel import compare_to_baseline, load_first_episodes, summarize_trace


def _row(env: int, step: int, *, done: bool = False, wrist: float = 0.02) -> dict:
    body_names = (
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    return {
        "env_id": env,
        "step_index": step,
        "rollout_index": 0,
        "motion_key": f"motion_{env}",
        "motion_time_step": step,
        "done": done,
        "termination_causes": ["motion_time_out"] if done else [],
        "anchor_pos_error_m": 0.01,
        "anchor_ori_error_rad": 0.02,
        "body_residuals": {
            name: {"norm_m": wrist if "wrist" in name else 0.01} for name in body_names
        },
        "structure_states": {"base_link": {"robot_pos_w_m": [0.0, 0.0, 0.65]}},
    }


def test_first_episode_excludes_post_reset_rows() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        rows = [_row(0, 0), _row(0, 1, done=True), _row(0, 2, wrist=1.0)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        episodes = load_first_episodes(path)
        assert len(episodes[0]) == 2
        summary = summarize_trace(path)
        assert summary["aggregate"]["completed"] == 1
        assert summary["aggregate"]["wrist_error_m"]["max"] == 0.02


def test_preservation_gate_detects_more_than_five_percent_wrist_regression() -> None:
    with tempfile.TemporaryDirectory() as directory:
        baseline_path = Path(directory) / "baseline.jsonl"
        candidate_path = Path(directory) / "candidate.jsonl"
        baseline_path.write_text("".join(json.dumps(_row(0, i, done=i == 2, wrist=0.02)) + "\n" for i in range(3)))
        candidate_path.write_text("".join(json.dumps(_row(0, i, done=i == 2, wrist=0.03)) + "\n" for i in range(3)))
        comparison = compare_to_baseline(summarize_trace(candidate_path), summarize_trace(baseline_path))
        assert comparison["relative_deltas"]["wrist_mean"] > 0.05
        assert comparison["upper_capability_preserved_5pct"] is False
