from pathlib import Path

import torch

from dcpeft_checkpoint_materialization import build_materialized_checkpoint
from gear_sonic.trl.utils.any2any_lora_checkpoint import merge_any2any_lora_state


def _lora_state():
    stem = "module.0"
    return {
        f"{stem}.lora_A": torch.tensor([[1.0, 2.0]]),
        f"{stem}.lora_B": torch.tensor([[3.0], [4.0]]),
        f"{stem}._lora_scaling": torch.tensor(0.5),
        f"{stem}._lora_output_mask": torch.tensor([1.0, 0.0]),
        f"{stem}._frozen_lora_delta_weight": torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]]
        ),
        f"{stem}.base_layer.weight": torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]]
        ),
        f"{stem}.base_layer.bias": torch.tensor([0.5, -0.5]),
        "other": torch.tensor([7.0]),
    }


def test_materializes_lora_and_omits_training_state():
    source = {
        "policy_state_dict": _lora_state(),
        "value_state_dict": {"value.weight": torch.tensor([[2.0]])},
        "optimizer_state_dict": {"should": "not survive"},
    }
    output, report = build_materialized_checkpoint(
        source,
        source_path=Path("/tmp/source.pt"),
        source_sha256="abc123",
        merge_state=merge_any2any_lora_state,
    )

    expected = torch.tensor([[11.6, 23.2], [30.3, 40.4]])
    torch.testing.assert_close(output["policy_state_dict"]["module.0.weight"], expected)
    torch.testing.assert_close(
        output["policy_state_dict"]["module.0.bias"], torch.tensor([0.5, -0.5])
    )
    assert "optimizer_state_dict" not in output
    assert report["states"]["policy_state_dict"]["merged_layer_count"] == 1
    assert not any(
        "lora" in key or "base_layer" in key
        for key in output["policy_state_dict"]
    )


def test_rejects_missing_required_state():
    try:
        build_materialized_checkpoint(
            {"policy_state_dict": _lora_state()},
            source_path=Path("/tmp/source.pt"),
            source_sha256="abc123",
            merge_state=merge_any2any_lora_state,
        )
    except KeyError as error:
        assert "value_state_dict" in str(error)
    else:
        raise AssertionError("missing value_state_dict should fail closed")
