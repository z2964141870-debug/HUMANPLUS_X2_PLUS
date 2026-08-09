#!/usr/bin/env python3
"""Expand the successful Phase3 walk smoke to the official five-action panel.

The walk-smoke teacher is reused and re-evaluated after the action-delta metric
fix.  The four remaining actions receive independent bounded CEM searches with
the same offline/free-root/prescribed-root contract.  No training or policy is
involved; all selected teachers remain full-future, simulator-only oracles.
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
import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as core


ROLES = ("walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand")
SMOKE_JSON = core.DEFAULT_JSON
SMOKE_CACHE = core.DEFAULT_CACHE


def baseline_entry(entry: dict[str, Any], model: mujoco.MjModel, phase2) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    phase = phase2.phase_contract(phase2.reference_kinematics(model, entry))
    result["phase3_expected_contact"] = {
        side: (~phase[f"{side}_swing"]).astype(np.uint8) for side in ("left", "right")
    }
    return result


def decision_for(
    baseline: dict[str, Any], free: dict[str, Any], prescribed: dict[str, Any]
) -> dict[str, Any]:
    gain = free["simulated_duration_s"] - baseline["simulated_duration_s"]
    slip_limit = max(1.10 * baseline["stance_slip_p95_mps"], baseline["stance_slip_p95_mps"] + 0.01)
    checks = {
        "survival_gain_at_least_0p5s": gain >= 0.50,
        "stance_slip_not_worse": free["stance_slip_p95_mps"] <= slip_limit,
        "prescribed_root_smooth_and_unsaturated": (
            prescribed["action_target_delta_p95_rad"] <= 0.22
            and prescribed["torque_saturation_fraction"] <= 0.03
        ),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "survival_gain_s": gain,
        "slip_limit_mps": slip_limit,
    }


def search_role(
    role: str,
    key: str,
    entry: dict[str, Any],
    model: mujoco.MjModel,
    phase2,
    physics,
    seed: int,
    population_count: int,
    iterations: int,
    elite_count: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    baseline = baseline_entry(entry, model, phase2)
    baseline_free = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, baseline, "free_root_balance", physics, phase2
    )
    rng = np.random.default_rng(seed)
    mean = (core.LOW + core.HIGH) / 2.0
    mean[9] = 0.065
    std = (core.HIGH - core.LOW) / 3.0
    best = None
    generated = offline_pass = rollouts = 0
    offline_failures: dict[str, int] = {}
    summaries = []
    finalists = []
    for iteration in range(iterations):
        population = rng.normal(mean, std, size=(population_count, len(mean)))
        population[0] = mean
        scored = []
        for raw in population:
            generated += 1
            vector = core.decode_vector(raw)
            teacher, diagnostics = core.build_teacher_entry(model, entry, vector, phase2)
            gate = core.offline_gate(teacher, diagnostics)
            if not gate["pass"]:
                for check, passed in gate["checks"].items():
                    if not passed:
                        offline_failures[check] = offline_failures.get(check, 0) + 1
                scored.append((1.0e6 + gate["joint_step_max_rad"], raw, None, teacher, vector, diagnostics, gate))
                continue
            offline_pass += 1
            free = core.simulate_dynamic_teacher(
                physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, teacher, "free_root_balance", physics, phase2
            )
            rollouts += 1
            cost = core.dynamic_cost(free)
            item = (cost, raw, free, teacher, vector, diagnostics, gate)
            scored.append(item)
            finalists.append({
                "cost": cost,
                "vector": core.asdict(vector),
                "diagnostics": diagnostics,
                "offline_gate": gate,
                "free_root": free,
            })
            if best is None or cost < best[0]:
                best = item
        scored.sort(key=lambda item: item[0])
        finite = [item for item in scored if item[0] < 1.0e6]
        if not finite:
            summaries.append({"iteration": iteration, "offline_pass_count": 0, "stopped": True})
            break
        elites = finite[: min(elite_count, len(finite))]
        elite_values = np.asarray([item[1] for item in elites])
        mean = 0.25 * mean + 0.75 * np.mean(elite_values, axis=0)
        std = np.maximum(0.15 * (core.HIGH - core.LOW), 0.25 * std + 0.75 * np.std(elite_values, axis=0))
        summaries.append({
            "iteration": iteration,
            "best_cost": float(elites[0][0]),
            "best_survival_s": elites[0][2]["simulated_duration_s"],
            "best_slip_p95_mps": elites[0][2]["stance_slip_p95_mps"],
            "offline_pass_count": len(finite),
        })
        print(
            f"[{role}] iteration={iteration} pass={len(finite)}/{population_count} "
            f"survival={elites[0][2]['simulated_duration_s']:.3f}s "
            f"slip={elites[0][2]['stance_slip_p95_mps']:.4f}",
            flush=True,
        )

    common = {
        "role": role,
        "motion_key": key,
        "search": {
            "seed": seed,
            "population": population_count,
            "iterations": iterations,
            "elite_count": elite_count,
            "generated_count": generated,
            "offline_pass_count": offline_pass,
            "free_root_rollout_count": rollouts,
            "offline_failure_counts": offline_failures,
            "iteration_summaries": summaries,
        },
        "baseline_free_root": baseline_free,
    }
    if best is None:
        common.update({
            "teacher_found": False,
            "decision": {
                "pass": False,
                "checks": {"offline_teacher_found": False},
                "survival_gain_s": None,
                "slip_limit_mps": None,
            },
        })
        return common, None

    cost, _, free, teacher, vector, diagnostics, gate = best
    prescribed = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE,
        physics.DEFAULT_CONTROL,
        teacher,
        "prescribed_root_trackability",
        physics,
        phase2,
    )
    decision = decision_for(baseline_free, free, prescribed)
    common.update({
        "teacher_found": True,
        "best_teacher_vector": core.asdict(vector),
        "best_teacher_diagnostics": diagnostics,
        "best_teacher_offline_gate": gate,
        "best_teacher_cost": cost,
        "best_teacher_free_root": free,
        "best_teacher_prescribed_root": prescribed,
        "decision": decision,
        "top_candidates": sorted(finalists, key=lambda item: item["cost"])[:5],
    })
    return common, teacher


def smoke_result(
    prior: dict[str, Any], teacher: dict[str, Any], entry: dict[str, Any], model, phase2, physics
) -> dict[str, Any]:
    baseline = baseline_entry(entry, model, phase2)
    baseline_free = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, baseline, "free_root_balance", physics, phase2
    )
    free = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, teacher, "free_root_balance", physics, phase2
    )
    prescribed = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, teacher, "prescribed_root_trackability", physics, phase2
    )
    return {
        "role": "walk_straight",
        "motion_key": prior.get("smoke_motion", {}).get("motion_key", prior.get("motion_key")),
        "search": prior["search"],
        "selection_origin": "reused successful 18x4 smoke; all physics metrics re-evaluated after action-delta fix",
        "teacher_found": True,
        "baseline_free_root": baseline_free,
        "best_teacher_vector": prior["best_teacher_vector"],
        "best_teacher_diagnostics": prior["best_teacher_diagnostics"],
        "best_teacher_offline_gate": prior["best_teacher_offline_gate"],
        "best_teacher_cost": core.dynamic_cost(free),
        "best_teacher_free_root": free,
        "best_teacher_prescribed_root": prescribed,
        "decision": decision_for(baseline_free, free, prescribed),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 Contact-Constrained Dynamic Teacher Phase3",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- walk smoke 先独立通过后才扩展四动作；全程无训练、无 checkpoint、无真机。",
        "- COM、DCM、contact、支撑相位均为官方 AimDK v1.0 MuJoCo 模型估计，并非真实 GRF/COP/COM/contact。",
        "- 所有 teacher 都使用完整未来与 free-root 物理选参，是 existence oracle，不可部署。",
        "",
        "## 可证伪契约",
        "",
        "优化变量覆盖 COM 横向/root 高度样条、接触切换、stance terminal 零速约束、swing clearance 与 landing terminal；free-root 代价覆盖生存、root tilt/DCM、slip、接触一致、action/torque 平滑。单动作晋升要求 `+0.5 s` 生存、slip 不超过容差且 prescribed-root 平滑/未饱和。",
        "",
        "## 五动作结果",
        "",
        "| role | generated/pass/rollout | baseline(s) | teacher(s) | gain(s) | baseline/teacher slip | action Δ | sat. | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["per_role"][role]
        search = value["search"]
        baseline = value["baseline_free_root"]
        if not value["teacher_found"]:
            lines.append(
                f"| {role} | {search['generated_count']}/{search['offline_pass_count']}/{search['free_root_rollout_count']} | "
                f"{baseline['simulated_duration_s']:.3f} | n/a | n/a | {baseline['stance_slip_p95_mps']:.4f}/n/a | n/a | n/a | False |"
            )
            continue
        teacher = value["best_teacher_free_root"]
        prescribed = value["best_teacher_prescribed_root"]
        lines.append(
            f"| {role} | {search['generated_count']}/{search['offline_pass_count']}/{search['free_root_rollout_count']} | "
            f"{baseline['simulated_duration_s']:.3f} | {teacher['simulated_duration_s']:.3f} | "
            f"{value['decision']['survival_gain_s']:.3f} | {baseline['stance_slip_p95_mps']:.4f}/{teacher['stance_slip_p95_mps']:.4f} | "
            f"{prescribed['action_target_delta_p95_rad']:.4f} | {prescribed['torque_saturation_fraction']:.4f} | {value['decision']['pass']} |"
        )
    lines += [
        "",
        "## 方法学自查",
        "",
    ]
    audit = report.get("methodology_audit")
    if audit:
        lines += [
            f"- walk teacher 证据：{audit['walk_evidence_boundary']}。",
            f"- 初始状态契约：{audit['initial_state_contract']}。",
            f"- 接触契约：{audit['contact_contract_interpretation']}。",
            f"- 表达能力限制：{audit['representation_limit']}。",
            "",
            "| role | first swing(s) | baseline fall(s) | fall before event | contact match | qdot0 p95/max(rad/s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for role in ROLES:
            value = audit["per_role"][role]
            lines.append(
                f"| {role} | {value['first_swing_time_s']:.3f} | {value['baseline_fall_time_s']:.3f} | "
                f"{value['fall_before_first_swing']} | {value['baseline_contact_match_fraction']:.3f} | "
                f"{value['frame0_joint_velocity_p95_radps']:.3f}/{value['frame0_joint_velocity_max_radps']:.3f} |"
            )
    lines += [
        "",
        "## 结果 / 结论 / 下一步",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def teacher_vector_from_dict(raw: dict[str, Any]) -> core.TeacherVector:
    return core.TeacherVector(
        com_lateral_knots_m=tuple(raw["com_lateral_knots_m"]),
        root_height_knots_m=tuple(raw["root_height_knots_m"]),
        liftoff_shift_frames=int(raw["liftoff_shift_frames"]),
        touchdown_shift_frames=int(raw["touchdown_shift_frames"]),
        swing_clearance_m=float(raw["swing_clearance_m"]),
        landing_dx_m=float(raw["landing_dx_m"]),
        landing_dy_m=float(raw["landing_dy_m"]),
    )


def add_methodology_audit(report: dict[str, Any], motions, role_to_key, model, phase2) -> None:
    rows = {}
    for role in ROLES:
        entry = motions[role_to_key[role]]
        phase = phase2.phase_contract(phase2.reference_kinematics(model, entry))
        _, start, _ = core.first_swing_event(phase)
        fps = float(entry["fps"])
        qdot0 = np.abs((np.asarray(entry["dof"])[1] - np.asarray(entry["dof"])[0]) * fps)
        baseline = report["per_role"][role]["baseline_free_root"]
        rows[role] = {
            "first_swing_time_s": start / fps,
            "baseline_fall_time_s": baseline["fall_time_s"],
            "fall_before_first_swing": bool(baseline["fall_time_s"] < start / fps),
            "baseline_contact_match_fraction": baseline["expected_contact_match_fraction_model_estimate"],
            "frame0_joint_velocity_p95_radps": float(np.percentile(qdot0, 95)),
            "frame0_joint_velocity_max_radps": float(np.max(qdot0)),
        }
    report["methodology_audit"] = {
        "walk_evidence_boundary": (
            "single clip, one CEM seed, deterministic official MuJoCo; no independent held-out clip, "
            "no multi-seed physics and still falls after the optimized short window"
        ),
        "initial_state_contract": (
            "qpos is exact reference frame 0 but qvel is forced to zero; nonzero frame-0 reference "
            "joint/root velocity is therefore not restored and can bias early failure"
        ),
        "contact_contract_interpretation": (
            "expected contact is inferred from model FK height/speed, not ground truth; 0.68-0.85 match "
            "on turn/start-stop actions indicates a likely schedule/official-contact mismatch"
        ),
        "representation_limit": (
            "the teacher optimizes COM lateral/root height and foot XYZ, but not root yaw, foot orientation, "
            "or centroidal angular momentum; turn_left failing terminal IK is not a clean dynamics impossibility proof"
        ),
        "per_role": rows,
    }


def audit_existing(args, phase2, physics) -> None:
    report = json.loads(args.json.read_text(encoding="utf-8"))
    motions = joblib.load(args.current)
    role_to_key = {entry["panel_role"]: key for key, entry in motions.items()}
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    role = "walk_straight"
    key = role_to_key[role]
    vector = teacher_vector_from_dict(report["per_role"][role]["best_teacher_vector"])
    teacher, diagnostics = core.build_teacher_entry(model, motions[key], vector, phase2)
    gate = core.offline_gate(teacher, diagnostics)
    if not gate["pass"]:
        raise RuntimeError("walk teacher no longer passes the final collocation generator offline gate")
    baseline = baseline_entry(motions[key], model, phase2)
    baseline_free = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, baseline, "free_root_balance", physics, phase2
    )
    free = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, teacher, "free_root_balance", physics, phase2
    )
    prescribed = core.simulate_dynamic_teacher(
        physics.DEFAULT_SCENE, physics.DEFAULT_CONTROL, teacher, "prescribed_root_trackability", physics, phase2
    )
    walk = report["per_role"][role]
    walk.update({
        "selection_origin": "original 18x4 vector rebuilt and re-evaluated under final 0.18-rad collocation constraint; no new search",
        "baseline_free_root": baseline_free,
        "best_teacher_diagnostics": diagnostics,
        "best_teacher_offline_gate": gate,
        "best_teacher_cost": core.dynamic_cost(free),
        "best_teacher_free_root": free,
        "best_teacher_prescribed_root": prescribed,
        "decision": decision_for(baseline_free, free, prescribed),
    })
    selected = joblib.load(args.cache)
    selected[key] = teacher
    joblib.dump(selected, args.cache)
    report["provenance"]["output_cache"] = {"path": str(args.cache), "sha256": core.sha256(args.cache)}
    add_methodology_audit(report, motions, role_to_key, model, phase2)
    pass_count = sum(value["decision"]["pass"] for value in report["per_role"].values())
    robust = pass_count >= 4
    report["decision"].update({
        "pass_count": pass_count,
        "robust_dynamic_teacher_exists": robust,
        "status": "PANEL_DYNAMIC_TEACHER_SUPPORTED" if robust else "PANEL_DYNAMIC_TEACHER_NOT_SUPPORTED",
        "result": f"{pass_count}/5 动作同时通过生存、滑移和 prescribed-root 门。",
        "conclusion": (
            "显式接触约束 dynamic teacher 在跨动作上成立，可进入因果化/held-out Silver 复核。"
            if robust
            else "walk 的单样本正结果未达到跨动作、held-out 或多seed稳健性，不能生成训练 Silver 集或解锁 PPO。"
        ),
        "next_step": (
            "冻结 oracle，做因果短窗口重现与 held-out 复核；仍不训练。"
            if robust
            else "停止扩大搜索；先修正初始qvel与接触事件/转向姿态契约，再重新预注册验证。"
        ),
    })
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=core.DEFAULT_CURRENT)
    parser.add_argument("--prior-smoke-json", type=Path, default=SMOKE_JSON)
    parser.add_argument("--prior-smoke-cache", type=Path, default=SMOKE_CACHE)
    parser.add_argument("--cache", type=Path, default=core.DEFAULT_CACHE)
    parser.add_argument("--json", type=Path, default=core.DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=core.DEFAULT_MD)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()

    phase2 = core.load_module(core.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase3_panel")
    physics = core.load_module(core.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase3_panel")
    if args.audit_existing:
        audit_existing(args, phase2, physics)
        return
    motions = joblib.load(args.current)
    role_to_key = {entry["panel_role"]: key for key, entry in motions.items()}
    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    prior = json.loads(args.prior_smoke_json.read_text(encoding="utf-8"))
    if "per_role" in prior:
        prior = prior["per_role"]["walk_straight"]
    prior_cache = joblib.load(args.prior_smoke_cache)
    smoke_key = role_to_key["walk_straight"]
    selected = {smoke_key: prior_cache[smoke_key]}
    per_role = {
        "walk_straight": smoke_result(
            prior, prior_cache[smoke_key], motions[smoke_key], model, phase2, physics
        )
    }
    for index, role in enumerate(ROLES[1:], start=1):
        key = role_to_key[role]
        print(f"[panel] search {role}", flush=True)
        result, teacher = search_role(
            role,
            key,
            motions[key],
            model,
            phase2,
            physics,
            seed=3407 + 101 * index,
            population_count=args.population,
            iterations=args.iterations,
            elite_count=args.elite_count,
        )
        per_role[role] = result
        if teacher is not None:
            selected[key] = teacher

    pass_count = sum(value["decision"]["pass"] for value in per_role.values())
    robust = pass_count >= 4
    report = {
        "schema_version": "x2_contact_constrained_dynamic_teacher_phase3_panel_v2",
        "provenance": {
            "current": {"path": str(args.current), "sha256": core.sha256(args.current)},
            "scene": {"path": str(physics.DEFAULT_SCENE), "sha256": core.sha256(physics.DEFAULT_SCENE)},
            "control": {"path": str(physics.DEFAULT_CONTROL), "sha256": core.sha256(physics.DEFAULT_CONTROL)},
            "prior_smoke_json_sha256": core.sha256(args.prior_smoke_json),
        },
        "truth_boundary": {
            "com_dcm_contact_source": "official AimDK v1.0 MuJoCo model estimate",
            "not_measured": ["hardware GRF", "hardware COP", "hardware COM", "real foot contact"],
            "oracle_deployable": False,
            "uses_full_future_reference": True,
            "training_or_checkpoint_load": False,
        },
        "expansion_contract": {
            "smoke_first": True,
            "smoke_required_survival_gain_s": 0.5,
            "panel_robust_pass_requirement": "at least 4 of 5 actions pass the identical per-action gates",
            "remaining_action_search": {
                "population": args.population, "iterations": args.iterations, "elite_count": args.elite_count
            },
        },
        "per_role": per_role,
        "decision": {
            "pass_count": pass_count,
            "motion_count": len(ROLES),
            "robust_dynamic_teacher_exists": robust,
            "status": "PANEL_DYNAMIC_TEACHER_SUPPORTED" if robust else "PANEL_DYNAMIC_TEACHER_NOT_SUPPORTED",
            "result": f"{pass_count}/5 动作同时通过生存、滑移和 prescribed-root 门。",
            "conclusion": (
                "显式接触约束 dynamic teacher 在跨动作上成立，可进入因果化/held-out Silver 复核。"
                if robust
                else "walk smoke 的正结果未达到跨动作稳健性，不能据此生成训练 Silver 集或解锁 PPO。"
            ),
            "next_step": (
                "冻结 oracle，做因果短窗口重现与 held-out 复核；仍不训练。"
                if robust
                else "分析未通过动作的接触事件/初始状态；不扩大 CEM 或五动作以外的数据。"
            ),
        },
    }
    add_methodology_audit(report, motions, role_to_key, model, phase2)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, args.cache)
    report["provenance"]["output_cache"] = {"path": str(args.cache), "sha256": core.sha256(args.cache)}
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
