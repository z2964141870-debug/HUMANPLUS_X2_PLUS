import json
from pathlib import Path

import torch

from tools.official_x2.export_rsl_actor_onnx import (
    ACTOR_INDICES,
    canonical_actor_state,
    checkpoint_actor_forward,
)


ROOT = Path(__file__).resolve().parents[1]


def _dense_actor_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(61)
    widths = (93, 256, 128, 128, 15)
    state = {}
    for layer, index in enumerate(ACTOR_INDICES):
        state[f"actor.{index}.weight"] = 0.05 * torch.randn(
            widths[layer + 1], widths[layer], generator=generator
        )
        state[f"actor.{index}.bias"] = 0.05 * torch.randn(
            widths[layer + 1], generator=generator
        )
    return state


def test_lora_merge_is_exact_for_batched_actor_forward():
    dense = _dense_actor_state()
    generator = torch.Generator().manual_seed(610)
    lora = {}
    for index in ACTOR_INDICES:
        weight = dense[f"actor.{index}.weight"]
        lora[f"actor.{index}.base.weight"] = weight
        lora[f"actor.{index}.base.bias"] = dense[f"actor.{index}.bias"]
        lora[f"actor.{index}.lora_A"] = 0.05 * torch.randn(
            4, weight.shape[1], generator=generator
        )
        lora[f"actor.{index}.lora_B"] = 0.05 * torch.randn(
            weight.shape[0], 4, generator=generator
        )
    merged, manifest = canonical_actor_state(lora, lora_alpha=4.0)
    canonical = {
        f"actor.{key}": value for key, value in merged.items()
    }
    sample = torch.randn(17, 93, generator=generator)
    torch.testing.assert_close(
        checkpoint_actor_forward(lora, sample, lora_alpha=4.0),
        checkpoint_actor_forward(canonical, sample, lora_alpha=None),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert manifest["lora_merged"] is True
    assert set(manifest["lora_rank_by_layer"].values()) == {4}


def test_dense_export_path_remains_unchanged():
    dense = _dense_actor_state()
    canonical, manifest = canonical_actor_state(dense, lora_alpha=None)
    assert manifest == {"source_format": "dense", "lora_merged": False}
    for index in ACTOR_INDICES:
        torch.testing.assert_close(
            canonical[f"{index}.weight"], dense[f"actor.{index}.weight"]
        )


def test_phase61_prereg_and_runner_freeze_complete_event():
    prereg = json.loads(
        (ROOT / "reports/retarget/x2_native_transition_posture_phase61_prereg.json").read_text()
    )
    assert prereg["event"]["rollout_s"] == 10.24
    assert prereg["event"]["terminal_in_double_support"] is True
    assert prereg["training"]["candidate_count"] == 1
    assert prereg["training"]["steps_per_env"] == 512
    assert prereg["reward"]["weights_are_reused_not_swept"] is True
    assert prereg["deployment_truth_boundary"]["local_pass_is_task2_completion"] is False

    runner = (ROOT / "scripts/run_x2_upper_robust_one_update_phase56.py").read_text()
    for token in (
        'CWI_PHASE61_TRANSITION_POSTURE',
        'TRAIN_STEPS = 512 if PEFT_PHASE61 else 24',
        'stand_s=0.0',
        'cruise_s=4.2',
        'decelerate_s=2.0',
        'terminal_double_support_penalty',
        '"terminal_zero_steps"',
    ):
        assert token in runner
