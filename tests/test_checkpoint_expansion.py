from collections import OrderedDict

import torch

from tools.expand_single_critic_checkpoint import (
    MASK_KEY,
    SCALED_OUTPUT_KEYS,
    expand_value_state,
)


def _state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "unchanged": torch.tensor([3.0, 4.0]),
            "critic_module.module.12.base_layer.weight": torch.arange(6.0).reshape(1, 6),
            "critic_module.module.12.base_layer.bias": torch.tensor([2.0]),
            "critic_module.module.12.lora_A": torch.arange(18.0).reshape(3, 6),
            "critic_module.module.12.lora_B": torch.tensor([[1.0, 2.0, 3.0]]),
            "critic_module.module.12._lora_output_mask": torch.tensor([1.0]),
        }
    )


def test_equal_split_preserves_aggregate_head() -> None:
    source = _state()
    dual = expand_value_state(source, 2)

    assert torch.equal(dual["unchanged"], source["unchanged"])
    assert torch.equal(
        dual["critic_module.module.12.lora_A"],
        source["critic_module.module.12.lora_A"],
    )
    for key in SCALED_OUTPUT_KEYS:
        assert dual[key].shape[0] == 2
        assert torch.equal(dual[key].sum(dim=0), source[key].squeeze(0))
    assert torch.equal(dual[MASK_KEY], torch.ones(2))


def test_source_is_not_mutated() -> None:
    source = _state()
    snapshot = OrderedDict((key, value.clone()) for key, value in source.items())
    _ = expand_value_state(source, 2)
    for key in source:
        assert torch.equal(source[key], snapshot[key])


def test_rejects_non_scalar_source_head() -> None:
    source = _state()
    source["critic_module.module.12.base_layer.bias"] = torch.ones(2)
    try:
        expand_value_state(source, 2)
    except ValueError as error:
        assert "not a scalar-head tensor" in str(error)
    else:
        raise AssertionError("non-scalar source head was accepted")
