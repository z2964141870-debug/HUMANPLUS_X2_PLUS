"""Minimal state adapter for Phase10 X2 native Gold MotionLib clips.

SONIC's native MotionLib consumes pose/root/fps correctly, but deliberately
recomputes velocities and contact labels from FK.  Phase10 also carries actual
official-simulation dq/root velocity and active-sole model contact.  This
adapter restores only those state-rich auxiliary fields in memory; it never
changes source PKL, pose, root, clip boundaries, or FPS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


def _ordered_entries(motion_lib: Any, source_pkl: Path) -> list[dict[str, Any]]:
    source = joblib.load(source_pkl)
    keys = list(motion_lib.curr_motion_keys)
    if keys != list(source.keys()):
        raise ValueError(f"MotionLib/source key order differs: {keys} vs {list(source.keys())}")
    entries = [source[key] for key in keys]
    frames = [len(entry["dof"]) for entry in entries]
    loaded_frames = [int(value) for value in motion_lib._motion_num_frames.cpu().tolist()]
    if frames != loaded_frames:
        raise ValueError(f"MotionLib/source frame boundaries differ: {loaded_frames} vs {frames}")
    return entries


def _cat(entries: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(entry[key]) for entry in entries], axis=0)


def apply_recorded_state_adapter(motion_lib: Any, source_pkl: Path) -> dict[str, Any]:
    """Restore actual dq/root velocities/contact without altering native FK state."""
    entries = _ordered_entries(motion_lib, source_pkl)
    before = {
        "dof_pos": motion_lib.dof_pos.detach().clone(),
        "body_pos_w": motion_lib.body_pos_w.detach().clone(),
        "body_quat_w": motion_lib.body_quat_w.detach().clone(),
    }
    device = motion_lib.dof_pos.device
    dtype = motion_lib.dof_pos.dtype
    dof_vel = torch.as_tensor(_cat(entries, "dof_vel"), device=device, dtype=dtype)
    root_lin = torch.as_tensor(_cat(entries, "root_lin_vel_w_mps"), device=device, dtype=dtype)
    root_ang = torch.as_tensor(_cat(entries, "root_ang_vel"), device=device, dtype=dtype)
    left = torch.as_tensor(
        np.concatenate([np.asarray(entry["model_contact"]["left"]) for entry in entries]),
        device=device,
        dtype=torch.bool,
    ).unsqueeze(-1)
    right = torch.as_tensor(
        np.concatenate([np.asarray(entry["model_contact"]["right"]) for entry in entries]),
        device=device,
        dtype=torch.bool,
    ).unsqueeze(-1)
    if tuple(dof_vel.shape) != tuple(motion_lib.dof_vel.shape):
        raise ValueError(f"recorded/derived dq shape differs: {dof_vel.shape} vs {motion_lib.dof_vel.shape}")
    motion_lib.dof_vel = dof_vel
    motion_lib.body_lin_vel_w[:, 0, :] = root_lin
    motion_lib.body_ang_vel_w[:, 0, :] = root_ang
    motion_lib.feet_l = left
    motion_lib.feet_r = right
    unchanged = {
        "dof_pos_exact": bool(torch.equal(before["dof_pos"], motion_lib.dof_pos)),
        "body_pos_exact": bool(torch.equal(before["body_pos_w"], motion_lib.body_pos_w)),
        "body_quat_exact": bool(torch.equal(before["body_quat_w"], motion_lib.body_quat_w)),
    }
    if not all(unchanged.values()):
        raise AssertionError(f"state adapter modified pose/FK fields: {unchanged}")
    return {
        "source_pkl": str(source_pkl),
        "frames": int(sum(len(entry["dof"]) for entry in entries)),
        "restored_fields": ["dof_vel", "root_lin_vel_w_mps", "root_ang_vel", "model_contact"],
        "unchanged_fields": unchanged,
    }


def wbt29_indices(official_names: list[str], wbt_names: list[str]) -> np.ndarray:
    if len(official_names) != 31 or len(wbt_names) != 29:
        raise ValueError("expected official31 and WBT29 names")
    if len(set(official_names)) != 31 or not set(wbt_names).issubset(official_names):
        raise ValueError("invalid official31/WBT29 name contract")
    return np.asarray([official_names.index(name) for name in wbt_names], dtype=np.int64)


def select_wbt29(values31: torch.Tensor, official_names: list[str], wbt_names: list[str]) -> torch.Tensor:
    indices = torch.as_tensor(wbt29_indices(official_names, wbt_names), device=values31.device)
    return values31.index_select(-1, indices)
