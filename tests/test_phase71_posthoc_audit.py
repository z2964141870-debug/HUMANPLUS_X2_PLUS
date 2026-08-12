import hashlib

import torch

from tools.retarget.audit_x2_phase71_posthoc import (
    cosine,
    prefixed_tensor_hash,
    raw_tensor_hash,
    relative_l2,
)


def test_rng_hash_conventions_are_distinct_and_reproducible() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(700042)
    state = generator.get_state()
    assert raw_tensor_hash(state) == "0721c6550d700ede33ea154260c940ab9d4be05b1d54bf192df7d51a89c668a8"
    assert prefixed_tensor_hash(state) == "408382543c7b8165648ae6e434929c29dc7f97601d49a0135c1a9ec1a1d0130f"
    assert raw_tensor_hash(state) != prefixed_tensor_hash(state)


def test_cuda_seed_state_can_be_audited_without_cuda() -> None:
    state = torch.tensor(
        list((700042).to_bytes(8, "little") + (0).to_bytes(8, "little")),
        dtype=torch.uint8,
    )
    assert raw_tensor_hash(state) == "2b6852ce608b9c5209ace6e7e6ca7006f4c8de288cbc5cc843a2a0bc3002e9a2"
    assert prefixed_tensor_hash(state) == "27b790394093efd7178dd2fc462278b20f3c5a49c8ec46bbe56ebae37a0c0d4f"


def test_vector_comparisons_use_float64() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert cosine(reference, reference) == 1.0
    assert relative_l2(reference, reference) == 0.0
    assert cosine(-reference, reference) == -1.0


def test_raw_hash_is_plain_bytes_sha() -> None:
    value = torch.tensor([1, 2, 3], dtype=torch.uint8)
    assert raw_tensor_hash(value) == hashlib.sha256(bytes([1, 2, 3])).hexdigest()
