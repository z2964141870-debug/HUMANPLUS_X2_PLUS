from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from x2_upper_robust_lora_phase57 import (
    dense_tensor_map,
    inject_standard93d_lora,
    tensor_map_hash,
)


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_upper_robust_lower_lora_live_zero_phase57.json"


class ToyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.std = nn.Parameter(torch.ones(15))
        self.actor = nn.Sequential(nn.Linear(93, 16), nn.ELU(), nn.Linear(16, 8), nn.ELU(), nn.Linear(8, 8), nn.ELU(), nn.Linear(8, 15))
        self.critic = nn.Sequential(nn.Linear(93, 16), nn.ELU(), nn.Linear(16, 8), nn.ELU(), nn.Linear(8, 8), nn.ELU(), nn.Linear(8, 1))


def test_phase57_lora_is_zero_exact_and_dense_frozen():
    torch.manual_seed(42)
    model = ToyActorCritic()
    inputs = torch.randn(7, 93)
    actor_before = model.actor(inputs).detach()
    critic_before = model.critic(inputs).detach()
    dense_before = tensor_map_hash(dense_tensor_map(model))
    manifest = inject_standard93d_lora(model)
    torch.testing.assert_close(model.actor(inputs), actor_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(model.critic(inputs), critic_before, atol=0.0, rtol=0.0)
    assert tensor_map_hash(dense_tensor_map(model)) == dense_before
    assert manifest["actor_scopes"] == ["actor.0", "actor.2", "actor.4", "actor.6"]
    assert manifest["critic_scopes"] == ["critic.0", "critic.2", "critic.4", "critic.6"]
    assert all("lora_" in name for name in manifest["trainable_names"])
    assert model.std.requires_grad is False


def test_phase57_live_report_is_exact_and_optimizer_free():
    result = json.loads(REPORT.read_text())
    assert result["decision"] == "PASS_LIVE_ZERO_UPDATE_ONLY"
    assert result["preregistered_learning_rate"] == 5.0e-5
    assert result["runtime"]["domain_upper_counts"] == {
        "none_ideal": 24, "none_response": 8,
        "bounded_ideal": 24, "bounded_response": 8,
    }
    assert result["boundary"]["policy_action_dim"] == 15
    assert result["boundary"]["legacy_body_partition"] == "29 = lower12+waist3 (15) + upper14"
    assert result["boundary"]["sim_partition"] == "31 = body29 + head2 nominal"
    assert result["dense_hash_before"] == result["dense_hash_after"]
    assert result["optimizer_constructed"] is False
    assert result["optimizer_steps"] == 0
    assert result["environment_control_steps"] == 0
    assert all(row["action_max_abs"] == row["value_max_abs"] == 0.0 for row in result["fixed_forward_batches"])
