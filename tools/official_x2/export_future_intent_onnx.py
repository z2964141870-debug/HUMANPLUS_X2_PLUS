#!/usr/bin/env python3
"""Export a Stage6 future-intent checkpoint as one deployable ONNX actor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from cwi_x2.future_intent_actor_critic import (
    BASE_ACTOR_OBS_DIM,
    COMMAND_OBS_START,
    DYNAMIC_INTENT_DIM,
    GAIT_PHASE_DIM,
    LOCOMOTION_INTENT_DIM,
    NUM_COORDINATION_MODES,
    NUM_LOWER_ACTIONS,
    RESPONSE_CONTEXT_DIM,
    RESPONSE_MODE_MASK,
    UPPER_INTENT_DIM,
    UPPER_ACTIVITY_DIM,
    build_coordination_basis,
    lower_response_context,
    upper_activity_features,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FutureIntentDeployActor(nn.Module):
    """Exact deterministic inference path of FutureIntentActorCritic."""

    def __init__(
        self,
        *,
        response_adapter_enabled: bool = False,
        transition_adapter_enabled: bool = False,
        transition_upper_conditioned: bool = False,
    ) -> None:
        super().__init__()
        self.response_adapter_enabled = bool(response_adapter_enabled)
        self.transition_adapter_enabled = bool(transition_adapter_enabled)
        self.transition_upper_conditioned = bool(transition_upper_conditioned)
        self.actor = nn.Sequential(
            nn.Linear(93, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 15),
        )
        self.coordination_adapter = nn.Sequential(
            nn.Linear(DYNAMIC_INTENT_DIM + GAIT_PHASE_DIM, 32),
            nn.ELU(),
            nn.Linear(32, NUM_COORDINATION_MODES),
        )
        if self.response_adapter_enabled:
            self.response_adapter = nn.Sequential(
                nn.Linear(
                    RESPONSE_CONTEXT_DIM + LOCOMOTION_INTENT_DIM + GAIT_PHASE_DIM,
                    32,
                ),
                nn.ELU(),
                nn.Linear(32, NUM_COORDINATION_MODES),
            )
        else:
            self.response_adapter = None
        if self.transition_adapter_enabled:
            self.transition_adapter = nn.Sequential(
                nn.Linear(
                    LOCOMOTION_INTENT_DIM
                    + GAIT_PHASE_DIM
                    + (UPPER_ACTIVITY_DIM if self.transition_upper_conditioned else 0),
                    16,
                ),
                nn.ELU(),
                nn.Linear(16, NUM_COORDINATION_MODES),
            )
        else:
            self.transition_adapter = None
        self.register_buffer("coordination_basis", build_coordination_basis())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base = obs[..., :BASE_ACTOR_OBS_DIM]
        intent = obs[..., BASE_ACTOR_OBS_DIM:]
        upper_intent = intent[..., :UPPER_INTENT_DIM]
        locomotion_intent = intent[..., UPPER_INTENT_DIM:]
        phase = base[..., -GAIT_PHASE_DIM:]
        coefficients = torch.tanh(
            self.coordination_adapter(torch.cat((intent, phase), dim=-1))
        )
        future_residual = torch.clamp(
            coefficients @ self.coordination_basis,
            min=-0.10,
            max=0.10,
        )
        gate = torch.clamp(
            torch.amax(torch.abs(upper_intent), dim=-1, keepdim=True) / 0.02,
            min=0.0,
            max=1.0,
        )
        forward_command = base[
            ...,
            COMMAND_OBS_START : COMMAND_OBS_START + 1,
        ]
        locomotion_gate = torch.clamp(
            torch.abs(forward_command) / 0.10,
            min=0.0,
            max=1.0,
        )
        future_residual = gate * locomotion_gate * future_residual
        response_residual = torch.zeros_like(future_residual)
        if self.response_adapter is not None:
            response_input = torch.cat(
                (lower_response_context(base), locomotion_intent, phase),
                dim=-1,
            )
            response_coefficients = torch.tanh(self.response_adapter(response_input))
            response_mode_mask = torch.as_tensor(
                RESPONSE_MODE_MASK,
                device=response_coefficients.device,
                dtype=response_coefficients.dtype,
            )
            response_coefficients = response_coefficients * response_mode_mask
            response_residual = torch.clamp(
                response_coefficients @ self.coordination_basis,
                min=-0.05,
                max=0.05,
            ) * locomotion_gate
        transition_residual = torch.zeros_like(future_residual)
        if self.transition_adapter is not None:
            transition_parts = [locomotion_intent]
            if self.transition_upper_conditioned:
                transition_parts.append(upper_activity_features(upper_intent))
            transition_parts.append(phase)
            transition_coefficients = torch.tanh(
                self.transition_adapter(torch.cat(transition_parts, dim=-1))
            )
            response_mode_mask = torch.as_tensor(
                RESPONSE_MODE_MASK,
                device=transition_coefficients.device,
                dtype=transition_coefficients.dtype,
            )
            transition_coefficients = transition_coefficients * response_mode_mask
            transition_gate = torch.clamp(
                torch.abs(locomotion_intent[..., 1:2]) / 0.10,
                min=0.0,
                max=1.0,
            )
            transition_residual = torch.clamp(
                transition_coefficients @ self.coordination_basis,
                min=-0.03,
                max=0.03,
            ) * transition_gate
        combined_residual = torch.clamp(
            future_residual + response_residual + transition_residual,
            min=-0.10,
            max=0.10,
        )
        return self.actor(base) + combined_residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    response_adapter_enabled = any(
        key.startswith("response_adapter.") for key in state
    )
    transition_adapter_enabled = any(
        key.startswith("transition_adapter.") for key in state
    )
    transition_upper_conditioned = bool(
        transition_adapter_enabled
        and state["transition_adapter.0.weight"].shape[1]
        == LOCOMOTION_INTENT_DIM + UPPER_ACTIVITY_DIM + GAIT_PHASE_DIM
    )
    model = FutureIntentDeployActor(
        response_adapter_enabled=response_adapter_enabled,
        transition_adapter_enabled=transition_adapter_enabled,
        transition_upper_conditioned=transition_upper_conditioned,
    )
    selected_prefixes = ["actor.", "coordination_adapter."]
    if response_adapter_enabled:
        selected_prefixes.append("response_adapter.")
    if transition_adapter_enabled:
        selected_prefixes.append("transition_adapter.")
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith(tuple(selected_prefixes))
    }
    current_state = model.state_dict()
    for name, prefix_dim in (
        ("coordination_adapter.0.weight", UPPER_INTENT_DIM),
        ("response_adapter.0.weight", RESPONSE_CONTEXT_DIM),
        ("transition_adapter.0.weight", LOCOMOTION_INTENT_DIM),
    ):
        if name not in selected:
            continue
        old_weight = selected[name]
        new_weight = current_state[name]
        if old_weight.shape == new_weight.shape:
            continue
        if old_weight.shape[1] + LOCOMOTION_INTENT_DIM == new_weight.shape[1]:
            migrated = torch.zeros_like(new_weight)
            migrated[:, :prefix_dim] = old_weight[:, :prefix_dim]
            migrated[:, prefix_dim + LOCOMOTION_INTENT_DIM :] = old_weight[
                :, prefix_dim:
            ]
            selected[name] = migrated
    expected = set(current_state) - {"coordination_basis"}
    if set(selected) != expected:
        raise RuntimeError(
            f"future-intent state mismatch: {sorted(set(selected) ^ expected)}"
        )
    model.load_state_dict(selected, strict=False)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.linspace(
        -0.2,
        0.2,
        BASE_ACTOR_OBS_DIM + DYNAMIC_INTENT_DIM,
        dtype=torch.float32,
    ).reshape(1, -1)
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    with torch.no_grad():
        torch_output = model(sample).numpy()
        zero_intent = sample.clone()
        zero_intent[:, BASE_ACTOR_OBS_DIM:] = 0.0
        zero_output = model(zero_intent)
        base_output = model.actor(zero_intent[:, :BASE_ACTOR_OBS_DIM])
        zero_intent_error = float(torch.max(torch.abs(zero_output - base_output)).item())
        zero_command = sample.clone()
        zero_command[
            :,
            COMMAND_OBS_START : COMMAND_OBS_START + 1,
        ] = 0.0
        zero_command[:, -LOCOMOTION_INTENT_DIM:] = 0.0
        stopped_output = model(zero_command)
        stopped_base_output = model.actor(zero_command[:, :BASE_ACTOR_OBS_DIM])
        zero_command_error = float(
            torch.max(torch.abs(stopped_output - stopped_base_output)).item()
        )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["actions"], {"obs": sample.numpy()})[0]
    max_error = float(np.max(np.abs(torch_output - onnx_output)))
    if (
        max_error > 2.0e-6
        or (not response_adapter_enabled and zero_intent_error != 0.0)
        or zero_command_error != 0.0
    ):
        raise RuntimeError(
            "future-intent export mismatch: "
            f"onnx={max_error} zero_intent={zero_intent_error} "
            f"zero_command={zero_command_error}"
        )

    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_iteration": int(payload.get("iter", -1)),
        "onnx": str(output),
        "onnx_sha256": sha256(output),
        "adapter_mode": "future",
        "response_adapter_enabled": response_adapter_enabled,
        "transition_adapter_enabled": transition_adapter_enabled,
        "transition_upper_conditioned": transition_upper_conditioned,
        "base_actor_frozen": True,
        "input": {
            "name": "obs",
            "shape": ["batch", BASE_ACTOR_OBS_DIM + DYNAMIC_INTENT_DIM],
            "dtype": "float32",
        },
        "output": {"name": "actions", "shape": ["batch", NUM_LOWER_ACTIONS], "dtype": "float32"},
        "dynamic_intent_contract": (
            "14 upper current deltas + 14 upper future deltas + normalized "
            "current vx and 1.0s future-minus-current vx"
        ),
        "coordination_output_abs_max": 0.10,
        "response_output_abs_max": 0.05 if response_adapter_enabled else 0.0,
        "response_context_contract": (
            "normalized lower q/dq plus previous 15-D action, locomotion intent, "
            "and gait phase"
            if response_adapter_enabled else None
        ),
        "transition_adapter_contract": (
            "2-D locomotion intent plus gait phase"
            + (" plus 2-D upper activity" if transition_upper_conditioned else "")
            + " to bounded no-yaw 8-mode residual"
            if transition_adapter_enabled else None
        ),
        "response_mode_contract": (
            "hip pitch/roll common+differential plus waist roll; no yaw correction"
            if response_adapter_enabled else None
        ),
        "locomotion_gate": "abs(vx) / 0.10; zero forward command returns exact base actor",
        "pytorch_onnx_max_abs_error": max_error,
        "zero_intent_base_actor_max_abs_error": zero_intent_error,
        "zero_command_base_actor_max_abs_error": zero_command_error,
    }
    output.with_suffix(".export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
