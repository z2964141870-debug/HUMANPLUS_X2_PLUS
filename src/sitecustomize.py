"""Import-time overlay for isolated reward-vector experiments.

Python imports this file automatically when ``src`` is on ``PYTHONPATH``.
Only when ``DCPEFT_REWARD_CONTRACT`` is set do we wrap ManagerEnvWrapper.step.
The source project is never modified.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
from types import ModuleType


_REWARD_TARGET = "gear_sonic.envs.wrapper.manager_env_wrapper"
_TRAINER_TARGET = "gear_sonic.trl.trainer.ppo_trainer"


def _patch_reward_wrapper(module: ModuleType) -> None:
    contract_path = os.environ.get("DCPEFT_REWARD_CONTRACT")
    if not contract_path:
        return
    if getattr(module.ManagerEnvWrapper, "_dcpeft_reward_contract_patched", False):
        return

    from dcpeft_reward_contract import load_contract
    import torch

    contract = load_contract(contract_path)
    audit_path = os.environ.get("DCPEFT_REWARD_AUDIT_JSONL")
    audit_limit = int(os.environ.get("DCPEFT_REWARD_AUDIT_LIMIT", "256"))
    original_step = module.ManagerEnvWrapper.step

    def contract_step(self, actions):
        result = original_step(self, actions)
        if len(result) != 4:
            raise RuntimeError(f"unexpected ManagerEnvWrapper.step result length: {len(result)}")
        obs, scalar_reward, dones, extras = result
        manager = self.env.reward_manager
        active_terms = tuple(manager.active_terms)
        contract.validate_terms(active_terms)
        term_to_index = {name: index for index, name in enumerate(active_terms)}
        # IsaacLab stores _step_reward without dt, whereas the scalar reward
        # includes dt. _reward_buf is the authoritative scalar for this step.
        step_dt = float(self.env.step_dt)
        per_term = manager._step_reward * step_dt
        grouped = torch.stack(
            [
                per_term[:, [term_to_index[name] for name in terms]].sum(dim=1)
                for terms in contract.group_terms
            ],
            dim=1,
        )
        reconstructed = grouped.sum(dim=1)
        scalar = scalar_reward.reshape(-1)
        max_abs_error = float((reconstructed - scalar).abs().max().item())
        if max_abs_error > contract.tolerance:
            raise RuntimeError(
                "reward contract equivalence failed: "
                f"contract={contract.contract_id}, max_abs_error={max_abs_error:.9g}, "
                f"tolerance={contract.tolerance:.9g}"
            )

        count = getattr(self, "_dcpeft_reward_audit_count", 0)
        if audit_path and count < audit_limit:
            record = {
                "step": count,
                "contract_id": contract.contract_id,
                "group_names": contract.group_names,
                "group_mean": grouped.mean(dim=0).detach().cpu().tolist(),
                "group_std": grouped.std(dim=0, unbiased=False).detach().cpu().tolist(),
                "scalar_mean": float(scalar.mean().item()),
                "max_abs_error": max_abs_error,
                "num_envs": int(grouped.shape[0]),
            }
            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._dcpeft_reward_audit_count = count + 1
        return obs, grouped, dones, extras

    module.ManagerEnvWrapper.step = contract_step
    module.ManagerEnvWrapper._dcpeft_reward_contract_patched = True


def _patch_trainer(module: ModuleType) -> None:
    """Add critic and actor-objective diagnostics without changing PPO math."""
    critic_diagnostics = os.environ.get("DCPEFT_CRITIC_DIAGNOSTICS", "0") == "1"
    actor_gradient_diagnostics = (
        os.environ.get("DCPEFT_ACTOR_GRADIENT_DIAGNOSTICS", "0") == "1"
    )
    if not critic_diagnostics and not actor_gradient_diagnostics:
        return
    trainer_class = module.TRLPPOTrainer
    if getattr(trainer_class, "_dcpeft_critic_diagnostics_patched", False):
        return

    import torch

    original_compute_returns = trainer_class._compute_returns
    original_compute_ppo_loss = trainer_class._compute_ppo_loss
    original_write_diagnostics = trainer_class._write_training_diagnostics
    original_adjust_learning_rate = trainer_class._adjust_learning_rate_based_on_kl
    original_create_optimizer = trainer_class.create_optimizer

    def diagnostic_compute_returns(self, values, last_values, policy_state_dict):
        returns, normalized_advantages = original_compute_returns(
            self, values, last_values, policy_state_dict
        )
        if not critic_diagnostics:
            return returns, normalized_advantages
        rewards = policy_state_dict["rewards"].to(values)
        dones = policy_state_dict["dones"].to(values.device).float()
        if values.ndim != 3 or rewards.shape != values.shape or returns.shape != values.shape:
            raise RuntimeError(
                "DC-PEFT critic shape contract failed: "
                f"values={tuple(values.shape)}, rewards={tuple(rewards.shape)}, "
                f"returns={tuple(returns.shape)}"
            )
        if values.shape[-1] != int(self.num_critics):
            raise RuntimeError(
                "DC-PEFT critic count mismatch: "
                f"tensor={values.shape[-1]}, config={self.num_critics}"
            )

        next_values = torch.cat((values[1:], last_values.unsqueeze(0)), dim=0)
        td_residual = rewards + (1.0 - dones) * float(self.gamma) * next_values - values
        raw_advantages = returns - values
        tensors = {
            "reward": rewards,
            "value": values,
            "return": returns,
            "advantage_raw": raw_advantages,
            "advantage_normalized": normalized_advantages,
            "td_residual": td_residual,
        }
        if not all(torch.isfinite(tensor).all() for tensor in tensors.values()):
            raise RuntimeError("DC-PEFT critic diagnostics found NaN/Inf")

        contract_path = os.environ.get("DCPEFT_REWARD_CONTRACT")
        if contract_path:
            from dcpeft_reward_contract import load_contract

            group_names = load_contract(contract_path).group_names
        else:
            group_names = tuple(f"scalar_{index}" for index in range(self.num_critics))
        if len(group_names) != self.num_critics:
            raise RuntimeError(
                f"reward groups {group_names} do not match num_critics={self.num_critics}"
            )

        metrics = {}
        reduce_dims = tuple(range(values.ndim - 1))
        for index, group_name in enumerate(group_names):
            prefix = f"dcpeft/critic_{index}_{group_name}"
            for tensor_name, tensor in tensors.items():
                head = tensor[..., index]
                metrics[f"{prefix}/{tensor_name}_mean"] = float(head.mean().item())
                metrics[f"{prefix}/{tensor_name}_std"] = float(
                    head.std(unbiased=False).item()
                )
            target = returns[..., index]
            residual = target - values[..., index]
            target_var = target.var(unbiased=False)
            explained_variance = (
                1.0 - residual.var(unbiased=False) / target_var
                if float(target_var.item()) > 1.0e-12
                else target_var.new_tensor(0.0)
            )
            metrics[f"{prefix}/explained_variance"] = float(explained_variance.item())
            metrics[f"{prefix}/value_loss_unclipped_preupdate"] = float(
                (0.5 * residual.square().mean()).item()
            )
        metrics["dcpeft/reward_value_shape_heads"] = int(values.shape[-1])
        self._dcpeft_last_critic_metrics = metrics
        return returns, normalized_advantages

    def diagnostic_adjust_learning_rate(self, kl_mean, optimizer):
        if actor_gradient_diagnostics and (
            os.environ.get("DCPEFT_ACTOR_GRADIENT_NONMUTATING", "1") == "1"
        ):
            return None
        return original_adjust_learning_rate(self, kl_mean, optimizer)

    def diagnostic_create_optimizer(self):
        optimizer = original_create_optimizer(self)
        if actor_gradient_diagnostics and (
            os.environ.get("DCPEFT_ACTOR_GRADIENT_NONMUTATING", "1") == "1"
        ):
            if optimizer is None:
                optimizer = self.optimizer
            for group in optimizer.param_groups:
                group["lr"] = 0.0
        return optimizer

    def diagnostic_compute_ppo_loss(self, forward_results, mb_rollout_data):
        result = original_compute_ppo_loss(self, forward_results, mb_rollout_data)
        if not actor_gradient_diagnostics:
            return result

        from dcpeft_gradient_geometry import (
            clipped_head_policy_losses,
            gradient_geometry,
        )
        from dcpeft_reward_contract import load_contract

        output_path = os.environ.get("DCPEFT_ACTOR_GRADIENT_JSONL")
        contract_path = os.environ.get("DCPEFT_REWARD_CONTRACT")
        if not output_path or not contract_path:
            raise RuntimeError(
                "actor-gradient diagnostics require DCPEFT_ACTOR_GRADIENT_JSONL "
                "and DCPEFT_REWARD_CONTRACT"
            )
        group_names = tuple(load_contract(contract_path).group_names)
        head_losses, reconstruction_error, valid_steps = clipped_head_policy_losses(
            result["pg_losses"],
            result["pg_losses2"],
            mb_rollout_data["mb_padding_mask"],
        )
        if len(head_losses) != len(group_names) or len(head_losses) < 2:
            raise RuntimeError(
                f"actor-gradient head mismatch: losses={len(head_losses)}, "
                f"groups={group_names}"
            )
        if reconstruction_error > 1.0e-6:
            raise RuntimeError(
                "per-head actor loss does not reconstruct PPO policy loss: "
                f"error={reconstruction_error}"
            )

        named_parameters = [
            (name, parameter)
            for name, parameter in self.policy_model.named_parameters()
            if parameter.requires_grad
        ]
        if not named_parameters:
            raise RuntimeError("actor-gradient diagnostics found no trainable actor parameters")
        parameter_names = tuple(name for name, _ in named_parameters)
        parameter_shapes = tuple(tuple(parameter.shape) for _, parameter in named_parameters)

        nonmutating = os.environ.get("DCPEFT_ACTOR_GRADIENT_NONMUTATING", "1") == "1"
        if nonmutating:
            nonzero_lrs = [
                float(group["lr"])
                for group in self.optimizer.param_groups
                if float(group["lr"]) != 0.0
            ]
            if nonzero_lrs:
                raise RuntimeError(
                    "non-mutating actor-gradient probe requires zero optimizer LRs, "
                    f"got {nonzero_lrs}"
                )

        if not hasattr(self, "_dcpeft_actor_gradient_initial_actor"):
            self._dcpeft_actor_gradient_initial_actor = {
                name: parameter.detach().cpu().clone()
                for name, parameter in named_parameters
            }
            self._dcpeft_actor_gradient_probe_index = 0
        accumulator = getattr(self, "_dcpeft_actor_gradient_accumulator", None)
        if accumulator is None:
            accumulator = {
                "group_names": group_names,
                "parameter_names": parameter_names,
                "parameter_shapes": parameter_shapes,
                "gradient_sums": {
                    group_name: {
                        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
                        for name, parameter in named_parameters
                    }
                    for group_name in group_names
                },
                "valid_steps": {group_name: 0 for group_name in group_names},
                "microbatch_cosines": [],
                "microbatch_norms": {group_name: [] for group_name in group_names},
                "microbatches": 0,
                "max_policy_loss_reconstruction_error": 0.0,
            }
            self._dcpeft_actor_gradient_accumulator = accumulator
        elif (
            accumulator["group_names"] != group_names
            or accumulator["parameter_names"] != parameter_names
            or accumulator["parameter_shapes"] != parameter_shapes
        ):
            raise RuntimeError("actor-gradient objective or parameter layout changed")

        gradients_by_group = {}
        for group_name, loss in zip(group_names, head_losses, strict=True):
            gradients = torch.autograd.grad(
                loss,
                [parameter for _, parameter in named_parameters],
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            detached = {}
            for (name, parameter), gradient in zip(
                named_parameters, gradients, strict=True
            ):
                value = (
                    gradient.detach().double().cpu()
                    if gradient is not None
                    else torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
                )
                detached[name] = value
                accumulator["gradient_sums"][group_name][name] += value * valid_steps
            accumulator["valid_steps"][group_name] += valid_steps
            gradients_by_group[group_name] = torch.cat(
                [detached[name].reshape(-1) for name in parameter_names]
            )

        geometry = gradient_geometry(gradients_by_group)
        pair_key = next(iter(geometry["pairwise_cosines"]))
        accumulator["microbatch_cosines"].append(
            float(geometry["pairwise_cosines"][pair_key])
        )
        for group_name in group_names:
            accumulator["microbatch_norms"][group_name].append(
                float(geometry["gradient_norms"][group_name])
            )
        accumulator["microbatches"] += 1
        accumulator["max_policy_loss_reconstruction_error"] = max(
            accumulator["max_policy_loss_reconstruction_error"],
            reconstruction_error,
        )
        return result

    def _flush_actor_gradient_diagnostics(self):
        if not actor_gradient_diagnostics:
            return {}
        accumulator = getattr(self, "_dcpeft_actor_gradient_accumulator", None)
        if not accumulator or accumulator["microbatches"] <= 0:
            raise RuntimeError("actor-gradient diagnostics collected no microbatches")

        from dcpeft_gradient_geometry import (
            assert_finite_record,
            finite_summary,
            gradient_geometry,
        )

        group_names = accumulator["group_names"]
        parameter_names = accumulator["parameter_names"]
        averaged = {}
        for group_name in group_names:
            count = accumulator["valid_steps"][group_name]
            if count <= 0:
                raise RuntimeError(f"actor-gradient group {group_name} has no valid steps")
            averaged[group_name] = {
                name: accumulator["gradient_sums"][group_name][name] / count
                for name in parameter_names
            }
        aggregate_vectors = {
            group_name: torch.cat(
                [averaged[group_name][name].reshape(-1) for name in parameter_names]
            )
            for group_name in group_names
        }
        aggregate_geometry = gradient_geometry(aggregate_vectors)
        parameter_geometry = {}
        for name in parameter_names:
            parameter_geometry[name] = gradient_geometry(
                {group_name: averaged[group_name][name] for group_name in group_names}
            )

        current_actor = {
            name: parameter.detach().cpu()
            for name, parameter in self.policy_model.named_parameters()
            if parameter.requires_grad
        }
        if current_actor.keys() != self._dcpeft_actor_gradient_initial_actor.keys():
            raise RuntimeError("actor-gradient trainable parameter set changed")
        actor_delta = 0.0
        for name, initial in self._dcpeft_actor_gradient_initial_actor.items():
            current = current_actor[name]
            if not torch.equal(current, initial):
                actor_delta = max(
                    actor_delta,
                    float(torch.max(torch.abs(current - initial)).item()),
                )
        nonmutating = os.environ.get("DCPEFT_ACTOR_GRADIENT_NONMUTATING", "1") == "1"
        if nonmutating and actor_delta != 0.0:
            raise RuntimeError(
                "non-mutating actor-gradient probe changed actor parameters: "
                f"max_abs_delta={actor_delta}"
            )

        probe_index = int(self._dcpeft_actor_gradient_probe_index)
        record = {
            "schema_version": 1,
            "probe_index": probe_index,
            "checkpoint": os.environ.get("DCPEFT_ACTOR_GRADIENT_CHECKPOINT"),
            "contract": os.environ.get("DCPEFT_REWARD_CONTRACT"),
            "env_seed": int(
                getattr(getattr(self.env, "config", None), "seed", 0)
            ),
            "trainer_seed": int(getattr(self.args, "seed", 0)),
            "loss_definition": "per-head clipped PPO policy gradient; shared entropy excluded",
            "nonmutating": nonmutating,
            "group_names": list(group_names),
            "microbatches": int(accumulator["microbatches"]),
            "valid_steps": dict(accumulator["valid_steps"]),
            "parameter_count": len(parameter_names),
            "parameter_shapes": {
                name: list(shape)
                for name, shape in zip(
                    parameter_names, accumulator["parameter_shapes"], strict=True
                )
            },
            "aggregate_geometry": aggregate_geometry,
            "microbatch_cosine": finite_summary(
                accumulator["microbatch_cosines"]
            ),
            "microbatch_gradient_norms": {
                group_name: finite_summary(
                    accumulator["microbatch_norms"][group_name]
                )
                for group_name in group_names
            },
            "parameter_geometry": parameter_geometry,
            "max_policy_loss_reconstruction_error": float(
                accumulator["max_policy_loss_reconstruction_error"]
            ),
            "actor_parameter_max_abs_delta": actor_delta,
        }
        assert_finite_record(record)
        path = Path(os.environ["DCPEFT_ACTOR_GRADIENT_JSONL"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._dcpeft_actor_gradient_probe_index = probe_index + 1
        self._dcpeft_actor_gradient_accumulator = None

        pair_key = next(iter(aggregate_geometry["pairwise_cosines"]))
        return {
            "dcpeft/actor_gradient/aggregate_cosine": float(
                aggregate_geometry["pairwise_cosines"][pair_key]
            ),
            "dcpeft/actor_gradient/micro_cosine_mean": float(
                record["microbatch_cosine"]["mean"]
            ),
            "dcpeft/actor_gradient/micro_negative_fraction": float(
                record["microbatch_cosine"]["negative_fraction"]
            ),
            "dcpeft/actor_gradient/actor_parameter_max_abs_delta": actor_delta,
        }

    def diagnostic_write(self, metrics):
        merged = dict(metrics)
        merged.update(getattr(self, "_dcpeft_last_critic_metrics", {}))
        merged.update(_flush_actor_gradient_diagnostics(self))
        return original_write_diagnostics(self, merged)

    trainer_class._compute_returns = diagnostic_compute_returns
    trainer_class._compute_ppo_loss = diagnostic_compute_ppo_loss
    trainer_class._adjust_learning_rate_based_on_kl = diagnostic_adjust_learning_rate
    trainer_class.create_optimizer = diagnostic_create_optimizer
    trainer_class._write_training_diagnostics = diagnostic_write
    trainer_class._dcpeft_critic_diagnostics_patched = True


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, original_loader, patch_function):
        self.original_loader = original_loader
        self.patch_function = patch_function

    def create_module(self, spec):
        create = getattr(self.original_loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        self.patch_function(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, targets):
        self.targets = targets

    def find_spec(self, fullname, path, target=None):
        patch_function = self.targets.get(fullname)
        if patch_function is None:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _PatchLoader(spec.loader, patch_function)
        return spec


_targets = {}
if os.environ.get("DCPEFT_REWARD_CONTRACT"):
    _targets[_REWARD_TARGET] = _patch_reward_wrapper
if (
    os.environ.get("DCPEFT_CRITIC_DIAGNOSTICS", "0") == "1"
    or os.environ.get("DCPEFT_ACTOR_GRADIENT_DIAGNOSTICS", "0") == "1"
):
    _targets[_TRAINER_TARGET] = _patch_trainer
if _targets:
    sys.meta_path.insert(0, _PatchFinder(_targets))
