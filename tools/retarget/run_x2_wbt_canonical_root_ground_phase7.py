#!/usr/bin/env python3
"""Derive and test one robot-level X2 root-ground/contact contract.

The contract is derived only from the official AimDK v1.0 scene/x2.xml reset
state, its 24 active sole collision spheres, and the source generator's already
registered 9-frame smoothing window.  It never selects root-z with per-clip
free-root survival.

Only root z changes.  Original and canonical references are replayed with the
same official prescribed/free-root zero-update harness.  Contact, COM and DCM
are simulator-model estimates and are not hardware GRF/COP truth.  No policy,
teacher, checkpoint, training, real robot, BASE, Git, or cloud operation is
used.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_contract_phase4 as phase4
import retarget.run_x2_wbt_generation_contract_phase5 as phase5


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT = phase3.DEFAULT_CURRENT
DEFAULT_MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
DEFAULT_CACHE = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "bronze_kinematic/phase7_canonical_root_ground/x2_phase7_root_ground.pkl"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_wbt_canonical_root_ground_phase7.json"
DEFAULT_MD = REPO / "reports/retarget/x2_wbt_canonical_root_ground_phase7.md"
ROLES = phase5.ROLES

# This is inherited from current_v4_exact30's source generator, not searched.
GROUND_SMOOTH_WINDOW = 9
SURVIVAL_REGRESSION_LIMIT_S = 0.10
SLIP_RELATIVE_ALLOWANCE = 1.10
SLIP_ABSOLUTE_ALLOWANCE_MPS = 0.020


def centered_moving_average(values: np.ndarray, window: int = GROUND_SMOOTH_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 0 or window % 2 != 1:
        raise ValueError("ground smoothing window must be a positive odd integer")
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def official_reset_geometry(model: mujoco.MjModel, physics) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor, foot_geoms = active_sole_spheres(model)
    fromto = np.zeros(6, dtype=np.float64)
    per_side = {
        side: min(
            float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto))
            for geom in geoms
        )
        for side, geoms in foot_geoms.items()
    }
    radii = [float(model.geom_size[geom, 0]) for geoms in foot_geoms.values() for geom in geoms]
    if len(radii) != 24 or max(radii) - min(radii) > 1.0e-12:
        raise AssertionError("official X2 sole must contain 12 equal-radius spheres per foot")
    return {
        "root_qpos_xyz_m": data.qpos[:3].copy(),
        "root_qpos_quat_wxyz": data.qpos[3:7].copy(),
        "per_side_min_signed_clearance_m": per_side,
        "reset_clearance_m": float(min(per_side.values())),
        "sole_sphere_radius_m": radii[0],
        "sole_collision_sphere_count": len(radii),
        "left_right_clearance_abs_delta_m": float(abs(per_side["left"] - per_side["right"])),
    }


def active_sole_spheres(model: mujoco.MjModel) -> tuple[int, dict[str, list[int]]]:
    """Return only the 12 active collision spheres per foot.

    The older physics helper grouped every geom on ankle-roll, including one
    non-contact visual mesh.  That is valid for visualization, but invalid for
    a canonical signed-distance contract.
    """
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        raise ValueError("official scene has no floor geom")
    foot_geoms: dict[str, list[int]] = {"left": [], "right": []}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for side in foot_geoms:
            if (
                body_name == f"{side}_ankle_roll_link"
                and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
                and int(model.geom_contype[geom_id]) != 0
                and int(model.geom_conaffinity[geom_id]) != 0
            ):
                foot_geoms[side].append(geom_id)
    if any(len(values) != 12 for values in foot_geoms.values()):
        raise AssertionError(f"expected 12 active sole spheres per foot: {foot_geoms}")
    return floor, foot_geoms


def sphere_distance_series(model, entry, phase2) -> dict[str, np.ndarray]:
    names = list(entry["joint_names_mujoco"])
    qpos_addresses, _ = phase2.joint_addresses(model, names)
    floor, foot_geoms = active_sole_spheres(model)
    data = mujoco.MjData(model)
    fromto = np.zeros(6, dtype=np.float64)
    result = {side: np.zeros(len(entry["dof"]), dtype=np.float64) for side in foot_geoms}
    for frame in range(len(entry["dof"])):
        phase2.set_reference_state(
            model,
            data,
            entry["root_trans_offset"][frame],
            entry["root_rot"][frame],
            entry["dof"][frame],
            qpos_addresses,
        )
        for side, geoms in foot_geoms.items():
            result[side][frame] = min(
                float(mujoco.mj_geomDistance(model, data, floor, geom, 1.0, fromto))
                for geom in geoms
            )
    return result


def apply_canonical_root_ground(
    model: mujoco.MjModel, entry: dict[str, Any], reset: dict[str, Any], phase2, physics
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = sphere_distance_series(model, entry, phase2)
    lowest_before = np.minimum(before["left"], before["right"])
    exact_correction = reset["reset_clearance_m"] - lowest_before
    correction = centered_moving_average(exact_correction)
    result = copy.deepcopy(entry)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64).copy()
    root[:, 2] += correction
    result["root_trans_offset"] = root.astype(np.asarray(entry["root_trans_offset"]).dtype)
    result["phase7_canonical_root_ground"] = {
        "formula": "z'=z+MA9(c_reset-min_24_official_sole_signed_distance(q,root))",
        "reset_clearance_m": reset["reset_clearance_m"],
        "moving_average_window_frames": GROUND_SMOOTH_WINDOW,
        "window_provenance": "current_v4_exact30 source generator smooth_window=9",
        "per_clip_survival_used_to_select_root_z": False,
        "changed_fields": ["root_trans_offset[:,2]"],
        "correction_m": correction,
    }
    after = sphere_distance_series(model, result, phase2)
    return result, ground_metrics(before, after, correction, reset)


def ground_metrics(before, after, correction, reset) -> dict[str, Any]:
    reset_clearance = float(reset["reset_clearance_m"])
    radius = float(reset["sole_sphere_radius_m"])
    lowest_before = np.minimum(before["left"], before["right"])
    lowest_after = np.minimum(after["left"], after["right"])
    error = lowest_after - reset_clearance
    contact_threshold = reset_clearance + radius
    geometry_contact = {
        side: np.asarray(after[side] <= contact_threshold, dtype=bool) for side in ("left", "right")
    }
    checks = {
        # Four sphere radii is a geometry-derived Bronze tolerance, not a force criterion.
        "lowest_clearance_error_p95_le_4r": float(np.percentile(np.abs(error), 95)) <= 4.0 * radius,
        "deep_penetration_fraction_le_1pct": float(np.mean(lowest_after < -4.0 * radius)) <= 0.01,
        "correction_step_le_sphere_diameter": float(np.max(np.abs(np.diff(correction)))) <= 2.0 * radius,
    }
    return {
        "raw_lowest_signed_distance_p05_p50_p95_m": [
            float(np.percentile(lowest_before, q)) for q in (5, 50, 95)
        ],
        "canonical_lowest_signed_distance_p05_p50_p95_m": [
            float(np.percentile(lowest_after, q)) for q in (5, 50, 95)
        ],
        "canonical_lowest_clearance_error_abs_p95_m": float(np.percentile(np.abs(error), 95)),
        "canonical_deep_penetration_fraction": float(np.mean(lowest_after < -4.0 * radius)),
        "correction_min_max_m": [float(np.min(correction)), float(np.max(correction))],
        "correction_step_p95_max_m": [
            float(np.percentile(np.abs(np.diff(correction)), 95)),
            float(np.max(np.abs(np.diff(correction)))),
        ],
        "geometry_contact_threshold_m": contact_threshold,
        "geometry_contact_fraction": {
            side: float(np.mean(values)) for side, values in geometry_contact.items()
        },
        "geometry_contact": geometry_contact,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def unchanged_gate(original: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "dof_exact": np.array_equal(original["dof"], candidate["dof"]),
        "root_xy_exact": np.array_equal(
            np.asarray(original["root_trans_offset"])[:, :2],
            np.asarray(candidate["root_trans_offset"])[:, :2],
        ),
        "root_rotation_exact": np.array_equal(original["root_rot"], candidate["root_rot"]),
        "fps_exact": original["fps"] == candidate["fps"],
        "joint_order_exact": original["joint_names_mujoco"] == candidate["joint_names_mujoco"],
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def slip_limit(baseline: float) -> float:
    return max(SLIP_RELATIVE_ALLOWANCE * baseline, baseline + SLIP_ABSOLUTE_ALLOWANCE_MPS)


def paired_gate(original_free: dict[str, Any], candidate_free: dict[str, Any]) -> dict[str, Any]:
    survival_delta = candidate_free["simulated_duration_s"] - original_free["simulated_duration_s"]
    allowed_slip = slip_limit(original_free["stance_slip_p95_mps"])
    checks = {
        "survival_not_worse_gt_0p1s": survival_delta >= -SURVIVAL_REGRESSION_LIMIT_S,
        "slip_within_allowance": candidate_free["stance_slip_p95_mps"] <= allowed_slip,
    }
    return {
        "survival_delta_s": float(survival_delta),
        "slip_limit_mps": float(allowed_slip),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def replay(entry, physics, phase2) -> dict[str, Any]:
    official_contact, prescribed = phase5.extract_official_contact(entry, physics, phase2)
    free = phase4.simulate_contract_case(
        physics.DEFAULT_SCENE,
        physics.DEFAULT_CONTROL,
        entry,
        "free_root_balance",
        "zero",
        physics,
        phase2,
        expected_contact=official_contact,
        contact_activation="collision",
    )
    return {"official_contact": official_contact, "prescribed": prescribed, "free": free}


def render(report: dict[str, Any]) -> str:
    reset = report["official_robot_contract"]["reset_geometry"]
    lines = [
        "# X2 WBT Canonical Root-Ground/Contact Contract Phase7",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 回到 original source generator；没有 source-time、foot offset、teacher、policy、训练、checkpoint 或真机干预。",
        "- MuJoCo collision/contact、COM、DCM 是官方模型估计量，**不是实机 GRF、COP 或足底力真值**。",
        "",
        "## 官方机器人级常量与公式",
        "",
        f"- `x2.xml` reset root-z：`{reset['root_qpos_xyz_m'][2]:.6f}m`。",
        f"- 24 个官方足底碰撞球，半径：`{reset['sole_sphere_radius_m']:.6f}m`；reset 最低 signed clearance：`{reset['reset_clearance_m']:.9f}m`。",
        "- 唯一 root-z 公式：`z'=z+MA9(c_reset-min_24_official_sole_signed_distance(q,root))`。MA9 继承 source generator 已固定的 smooth-window，不由五条动作结果选择。",
        "- 模型 contact intent：单脚最低球距离 `<= c_reset + sphere_radius`；official realized contact 则来自 prescribed MuJoCo collision。二者都不是硬件力真值。",
        "",
        "## Phase4–6 符号分裂根因",
        "",
        "raw `gmr` 将不同 AMASS actor/sequence 的绝对 pelvis 高度带入 X2：KIT walk root-z 中位数约 0.695m，四条 ACCAD 约 0.614–0.626m；同一 X2 因而一条悬空、四条穿地。canonical 公式删除该绝对高度泄漏，但保留 joint、root XY/orientation 和 FPS。",
        "",
        "另发现 Phase4–6 signed-distance helper 把 ankle-roll 的 non-contact visual mesh 与 12 个 active sole spheres 混在一起；这污染旧几何距离数字，但不影响 MuJoCo 实际 collision replay。Phase7 已只保留每脚 12 个 active sphere。",
        "",
        "## 固定五动作 paired gate",
        "",
        "| role | raw root-z p50 | correction min..max / step max | ground err p95 | contact agree/F1 | original→canonical survival | Δ(s) | slip/limit | paired |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["per_role"][role]
        ground = value["ground"]
        gate = value["paired_gate"]
        original = value["physics"]["original"]["free"]
        candidate = value["physics"]["canonical"]["free"]
        lines.append(
            f"| {role} | {value['raw_root_z_p50_m']:.3f} | "
            f"{ground['correction_min_max_m'][0]:.3f}..{ground['correction_min_max_m'][1]:.3f}/"
            f"{ground['correction_step_p95_max_m'][1]:.4f} | "
            f"{ground['canonical_lowest_clearance_error_abs_p95_m']:.4f} | "
            f"{value['contact_comparison']['macro_agreement']:.3f}/"
            f"{value['contact_comparison']['macro_f1']:.3f} | "
            f"{original['simulated_duration_s']:.3f}→{candidate['simulated_duration_s']:.3f} | "
            f"{gate['survival_delta_s']:+.3f} | {candidate['stance_slip_p95_mps']:.4f}/"
            f"{gate['slip_limit_mps']:.4f} | {gate['pass']} |"
        )
    lines += [
        "",
        "## 裁决边界",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "- Bronze canonical 几何通过不等于 dynamic Silver，更不等于可训练或可部署。当前结果只否定把该 open-loop reference 直接晋级 Silver；不否定 X2、Any2Any 或 WBT。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--model-contract", type=Path, default=DEFAULT_MODEL_CONTRACT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase7")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase7")
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    reset = official_reset_geometry(model, physics)
    model_contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    motions = joblib.load(args.current)
    role_to_key = {entry["panel_role"]: key for key, entry in motions.items()}
    names_31 = model_contract["control_boundaries"]["official_mjcf_actuated_31"]
    names_29 = model_contract["control_boundaries"]["official_rl_sample_29"]
    permutation = [names_31.index(name) for name in names_29]

    per_role: dict[str, Any] = {}
    output_cache: dict[str, Any] = {}
    for role in ROLES:
        print(f"[phase7] {role}", flush=True)
        key = role_to_key[role]
        original = motions[key]
        candidate, ground = apply_canonical_root_ground(model, original, reset, phase2, physics)
        unchanged = unchanged_gate(original, candidate)
        original_physics = replay(original, physics, phase2)
        candidate_physics = replay(candidate, physics, phase2)
        contact_comparison = phase4.compare_contact_labels(
            ground.pop("geometry_contact"), candidate_physics["official_contact"]
        )
        contact_realization_gate = {
            "threshold_provenance": "Phase4 registered macro agreement gate",
            "macro_agreement_min": 0.90,
            "pass": bool(contact_comparison["macro_agreement"] >= 0.90),
        }
        gate = paired_gate(original_physics["free"], candidate_physics["free"])
        output_cache[key] = candidate
        per_role[role] = {
            "motion_key": key,
            "source_dataset_family": "KIT" if "/KIT/" in original["source_smplx_file"] else "ACCAD",
            "raw_root_z_p05_p50_p95_m": [
                float(np.percentile(np.asarray(original["root_trans_offset"])[:, 2], q))
                for q in (5, 50, 95)
            ],
            "raw_root_z_p50_m": float(np.median(np.asarray(original["root_trans_offset"])[:, 2])),
            "ground": ground,
            "unchanged_gate": unchanged,
            "contact_comparison": contact_comparison,
            "contact_realization_gate": contact_realization_gate,
            "physics": {
                "original": {
                    "prescribed": original_physics["prescribed"],
                    "free": original_physics["free"],
                },
                "canonical": {
                    "prescribed": candidate_physics["prescribed"],
                    "free": candidate_physics["free"],
                },
            },
            "paired_gate": gate,
        }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_cache, args.cache)
    geometry_all = all(value["ground"]["pass"] for value in per_role.values())
    unchanged_all = all(value["unchanged_gate"]["pass"] for value in per_role.values())
    prescribed_all = all(
        value["physics"]["canonical"]["prescribed"]["duration_fraction"] >= 0.999
        for value in per_role.values()
    )
    survival_all = all(
        value["paired_gate"]["checks"]["survival_not_worse_gt_0p1s"] for value in per_role.values()
    )
    slip_all = all(
        value["paired_gate"]["checks"]["slip_within_allowance"] for value in per_role.values()
    )
    contact_all = all(value["contact_realization_gate"]["pass"] for value in per_role.values())
    dynamic_all = bool(
        geometry_all and unchanged_all and prescribed_all and survival_all and slip_all and contact_all
    )
    slip_failed = [
        role for role, value in per_role.items()
        if not value["paired_gate"]["checks"]["slip_within_allowance"]
    ]
    contact_failed = [
        role for role, value in per_role.items() if not value["contact_realization_gate"]["pass"]
    ]
    report = {
        "schema_version": "x2_wbt_canonical_root_ground_phase7_v1",
        "provenance": {
            "source": {"path": str(args.current), "sha256": phase3.sha256(args.current)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase3.sha256(physics.DEFAULT_SCENE)},
            "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": phase3.sha256(physics.DEFAULT_CONTROL)},
            "official_model_contract": {"path": str(args.model_contract), "sha256": phase3.sha256(args.model_contract)},
            "output_cache": {"path": str(args.cache), "sha256": phase3.sha256(args.cache)},
        },
        "truth_boundary": {
            "contact_source": "official MuJoCo collision plus geometry-derived near-ground intent",
            "contact_com_dcm_are_model_estimates": True,
            "not_hardware_grf_cop_or_foot_force": True,
            "training_teacher_checkpoint_real_robot": False,
        },
        "official_robot_contract": {
            "reset_geometry": {
                **reset,
                "root_qpos_xyz_m": reset["root_qpos_xyz_m"].tolist(),
                "root_qpos_quat_wxyz": reset["root_qpos_quat_wxyz"].tolist(),
            },
            "official_rl_29_to_mjcf_31_indices": permutation,
            "head_locked_31_indices": [names_31.index(name) for name in model_contract["control_boundaries"]["head_locked_2"]],
            "formula": "z'=z+MA9(c_reset-min_24_official_sole_signed_distance(q,root))",
            "root_z_selection_uses_per_clip_survival": False,
            "contact_intent_threshold": "c_reset + sole sphere radius",
        },
        "root_sign_audit": {
            "cause": "source actor/sequence absolute pelvis height leaks through root_z_mode=gmr",
            "walk_family": "KIT",
            "other_four_family": "ACCAD",
            "not_two_official_ground_planes": True,
            "phase4_6_distance_helper_included_visual_ankle_mesh": True,
            "actual_collision_replay_affected_by_visual_mesh": False,
            "phase7_fix": "distance/root-z audit uses only 12 active sole spheres per foot",
        },
        "per_role": per_role,
        "decision": {
            "checks": {
                "canonical_geometry_all_five": geometry_all,
                "only_root_z_changed_all_five": unchanged_all,
                "prescribed_full_all_five": prescribed_all,
                "free_survival_no_gt_0p1s_regression_all_five": survival_all,
                "free_slip_within_allowance_all_five": slip_all,
                "geometry_intent_matches_prescribed_realized_contact_all_five": contact_all,
                "training_or_teacher_executed": False,
            },
            "geometry_contract_bronze_valid": bool(geometry_all and unchanged_all),
            "dynamic_silver_promotable": dynamic_all,
            "slip_failed_roles": slip_failed,
            "contact_failed_roles": contact_failed,
            "status": (
                "PHASE7_CANONICAL_ROOT_GROUND_DYNAMIC_PROMOTABLE"
                if dynamic_all
                else "PHASE7_CANONICAL_GEOMETRY_VALID_DYNAMIC_NOT_PROMOTABLE"
            ),
            "result": (
                "五动作 geometry/prescribed/free survival/slip 全部通过。"
                if dynamic_all
                else (
                    f"五动作 root geometry 与 survival 通过；slip 在 {slip_failed} 越门，"
                    f"prescribed realized-contact agreement 在 {contact_failed} 未过 Phase4 0.90 门。"
                )
            ),
            "conclusion": (
                "机器人级 root-ground/contact contract 可晋级动态数据生成。"
                if dynamic_all
                else "统一公式解决 source root-z 符号分裂，可冻结为 Bronze 几何契约；contact realization 与当前 open-loop reference 均不能直接晋级 dynamic Silver。"
            ),
            "next_step": (
                "冻结 Phase7 contract，扩大诊断 panel 后再考虑 Silver。"
                if dynamic_all
                else "冻结公式与失败证据，不调 clearance/window；下一阶段应修 stance-foot 水平约束/接触时序，而非再改 root-z。"
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
