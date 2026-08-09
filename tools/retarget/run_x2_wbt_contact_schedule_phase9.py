#!/usr/bin/env python3
"""Freeze one collision-derived contact-label contract for the X2 WBT panel.

Phase7's robot-level root/ground Bronze motion is immutable here.  The only
intervention is an offline, robot-wide temporal contract applied to the
prescribed-root official-collision events already recorded by Phase7.  A
single deterministic free-root zero-update replay per motion is permitted only
to recover the contact trace that Phase7 did not persist.  Its aggregate
survival/slip must first reproduce Phase7; no second physics pass is run.

All contact labels are official-simulator collision estimates.  They are not
hardware GRF, COP, wrench, or force truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import retarget.run_x2_contact_constrained_dynamic_teacher_phase3 as phase3
import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7
import retarget.run_x2_wbt_contract_phase4 as phase4


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PHASE7 = phase7.DEFAULT_CACHE
DEFAULT_PHASE7_REPORT = phase7.DEFAULT_JSON
DEFAULT_JSON = REPO / "reports/retarget/x2_wbt_contact_schedule_phase9.json"
DEFAULT_MD = REPO / "reports/retarget/x2_wbt_contact_schedule_phase9.md"
ROLES = phase7.ROLES

# Pre-registered from the 30 Hz label resolution, not selected per clip:
# two frames confirm a transition (66.7 ms), while every retained state must
# persist for at least three frames (100 ms).
ON_CONFIRM_FRAMES = 2
OFF_CONFIRM_FRAMES = 2
MIN_DWELL_FRAMES = 3
REPLAY_METRIC_ATOL = 1.0e-9


def labels_from_events(frame_count: int, events: dict[str, list[int]]) -> np.ndarray:
    """Exactly reconstruct a Boolean series from Phase7 transition events."""
    touches = set(int(value) for value in events["touchdown"])
    lifts = set(int(value) for value in events["liftoff"])
    if touches & lifts:
        raise ValueError("touchdown and liftoff cannot share a frame")
    values = np.zeros(frame_count, dtype=bool)
    state = False
    for frame in range(frame_count):
        if frame in touches:
            if state:
                raise ValueError(f"duplicate touchdown at frame {frame}")
            state = True
        if frame in lifts:
            if not state:
                raise ValueError(f"liftoff without contact at frame {frame}")
            state = False
        values[frame] = state
    if phase4.event_frames(values) != {
        "touchdown": sorted(touches), "liftoff": sorted(lifts)
    }:
        raise AssertionError("event reconstruction is not lossless")
    return values


def run_intervals(values: np.ndarray) -> list[tuple[int, int, bool]]:
    values = np.asarray(values, dtype=bool)
    if not len(values):
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return [(int(start), int(end), bool(values[start])) for start, end in zip(starts, ends)]


def hysteresis_filter(
    values: np.ndarray,
    on_confirm: int = ON_CONFIRM_FRAMES,
    off_confirm: int = OFF_CONFIRM_FRAMES,
) -> np.ndarray:
    """Debounce a Boolean trace and backdate a confirmed edge to its first frame."""
    raw = np.asarray(values, dtype=bool)
    if not len(raw):
        return raw.copy()
    if on_confirm < 1 or off_confirm < 1:
        raise ValueError("confirmation lengths must be positive")
    result = np.full(len(raw), raw[0], dtype=bool)
    state = bool(raw[0])
    pending_start: int | None = None
    for frame in range(1, len(raw)):
        if bool(raw[frame]) == state:
            pending_start = None
            result[frame] = state
            continue
        if pending_start is None:
            pending_start = frame
        confirm = on_confirm if bool(raw[frame]) else off_confirm
        if frame - pending_start + 1 >= confirm:
            state = bool(raw[frame])
            result[pending_start : frame + 1] = state
            pending_start = None
        else:
            result[frame] = state
    return result


def enforce_min_dwell(values: np.ndarray, minimum: int = MIN_DWELL_FRAMES) -> np.ndarray:
    """Remove sub-minimum binary islands without clip-specific decisions."""
    if minimum < 1:
        raise ValueError("minimum dwell must be positive")
    result = np.asarray(values, dtype=bool).copy()
    for _ in range(len(result) + 1):
        intervals = run_intervals(result)
        short = [(start, end, state) for start, end, state in intervals if end - start < minimum]
        if not short or len(intervals) == 1:
            return result
        changed = False
        for index, (start, end, state) in enumerate(intervals):
            if end - start >= minimum:
                continue
            if index == 0:
                replacement = intervals[index + 1][2]
            elif index == len(intervals) - 1:
                replacement = intervals[index - 1][2]
            else:
                # A Boolean island's two neighbours necessarily share a state.
                if intervals[index - 1][2] != intervals[index + 1][2]:
                    raise AssertionError("binary island neighbours must agree")
                replacement = intervals[index - 1][2]
            result[start:end] = replacement
            changed = changed or replacement != state
        if not changed:
            return result
    raise RuntimeError("min-dwell cleanup did not converge")


def canonical_schedule(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        side: enforce_min_dwell(hysteresis_filter(np.asarray(raw[side], dtype=bool)))
        for side in ("left", "right")
    }


def phase_statistics(labels: dict[str, np.ndarray]) -> dict[str, Any]:
    left = np.asarray(labels["left"], dtype=bool)
    right = np.asarray(labels["right"], dtype=bool)
    if len(left) != len(right):
        raise ValueError("left/right label lengths differ")
    ds = left & right
    left_ss = left & ~right
    right_ss = right & ~left
    flight = ~left & ~right
    states = np.where(ds, 0, np.where(left_ss, 1, np.where(right_ss, 2, 3)))
    segments = []
    if len(states):
        starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
        ends = np.r_[starts[1:], len(states)]
        names = ("double_support", "left_single_support", "right_single_support", "flight")
        segments = [
            {"start": int(start), "end_exclusive": int(end), "state": names[int(states[start])]}
            for start, end in zip(starts, ends)
        ]
    ds_ss_ds = sum(
        1
        for index in range(1, len(segments) - 1)
        if segments[index - 1]["state"] == "double_support"
        and segments[index]["state"] in ("left_single_support", "right_single_support")
        and segments[index + 1]["state"] == "double_support"
    )
    total = max(1, len(left))
    return {
        "contact_window_count": {
            side: sum(state for _, _, state in run_intervals(labels[side]))
            for side in ("left", "right")
        },
        "double_support_ratio": float(np.sum(ds) / total),
        "single_support_ratio": float(np.sum(left_ss | right_ss) / total),
        "left_single_support_ratio": float(np.sum(left_ss) / total),
        "right_single_support_ratio": float(np.sum(right_ss) / total),
        "flight_ratio": float(np.sum(flight) / total),
        "ds_ss_ds_cycle_count": int(ds_ss_ds),
        "phase_segment_count": len(segments),
        "segments": segments,
    }


def event_error_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        result[side] = {}
        errors = comparison["per_side"][side]["nearest_event_error_frames"]
        for event in ("liftoff", "touchdown"):
            values = errors[event]
            result[side][event] = {
                "count": len(values),
                "mean_abs_frames": float(np.mean(values)) if values else None,
                "max_abs_frames": int(max(values)) if values else None,
            }
    return result


def array_sha256(labels: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for side in ("left", "right"):
        digest.update(np.asarray(labels[side], dtype=np.uint8).tobytes())
    return digest.hexdigest()


def replay_reproduced(current: dict[str, float], old: dict[str, float]) -> dict[str, Any]:
    delta = {
        "simulated_duration_s": float(current["simulated_duration_s"] - old["simulated_duration_s"]),
        "stance_slip_p95_mps": float(current["stance_slip_p95_mps"] - old["stance_slip_p95_mps"]),
    }
    checks = {name: abs(value) <= REPLAY_METRIC_ATOL for name, value in delta.items()}
    return {"delta": delta, "atol": REPLAY_METRIC_ATOL, "checks": checks, "pass": bool(all(checks.values()))}


def preservation_gate(raw: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "not_all_double_support": candidate["double_support_ratio"] < 1.0,
        "single_support_not_deleted": candidate["single_support_ratio"]
        >= 0.50 * raw["single_support_ratio"],
        "left_ss_preserved_if_present": raw["left_single_support_ratio"] == 0.0
        or candidate["left_single_support_ratio"] > 0.0,
        "right_ss_preserved_if_present": raw["right_single_support_ratio"] == 0.0
        or candidate["right_single_support_ratio"] > 0.0,
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def schedule_gate(
    source: dict[str, Any], candidate: dict[str, Any], preservation: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "agreement_strictly_improves": candidate["macro_agreement"] > source["macro_agreement"] + 1.0e-12,
        "macro_f1_not_worse": candidate["macro_f1"] + 1.0e-12 >= source["macro_f1"],
        "single_support_semantics_preserved": preservation["pass"],
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 WBT Collision Contact Schedule Phase9",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- Phase7 root-ground Bronze、动作、root、关节、控制和物理模型全部冻结；只处理 contact label。",
        "- 五动作各只补录一次相同 free-root zero-update contact trace；先复现 Phase7 aggregate，之后全部离线，未做第二轮 physics。",
        "- MuJoCo 路径无随机采样/seed；scene、control 与 Phase7 Bronze 均记录 SHA256。",
        "- collision/contact 是官方 MuJoCo 模型估计，**不是实机 GRF、COP、足底力或 wrench 真值**。",
        "",
        "## 预注册统一契约",
        "",
        f"- 30Hz 上 contact-on/off 均需连续 `{ON_CONFIRM_FRAMES}` 帧确认；保留状态最短 `{MIN_DWELL_FRAMES}` 帧。",
        "- 参数对五动作统一，不按 clip、free survival 或结果调节；schedule 不修改任何运动轨迹。",
        "- DS/左SS/右SS/flight 仅由两脚标签确定；禁止退化成全程双接触，且必须保留原始左右单支撑语义。",
        "",
        "## 共同存活窗结果",
        "",
        "| role | frames | raw windows L/R | schedule windows L/R | source agree/F1 | schedule agree/F1 | SS raw→schedule→free | event max source→schedule | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = report["per_role"][role]
        source = value["comparison_to_free_realized"]["phase7_source_intent"]
        candidate = value["comparison_to_free_realized"]["canonical_schedule"]
        raw_phase = value["phase_statistics"]["raw_prescribed"]
        candidate_phase = value["phase_statistics"]["canonical_schedule"]
        free_phase = value["phase_statistics"]["free_realized"]
        source_event = source["max_nearest_event_error_frames"]
        candidate_event = candidate["max_nearest_event_error_frames"]
        lines.append(
            f"| {role} | {value['common_survival_frames']} | "
            f"{raw_phase['contact_window_count']['left']}/{raw_phase['contact_window_count']['right']} | "
            f"{candidate_phase['contact_window_count']['left']}/{candidate_phase['contact_window_count']['right']} | "
            f"{source['macro_agreement']:.3f}/{source['macro_f1']:.3f} | "
            f"{candidate['macro_agreement']:.3f}/{candidate['macro_f1']:.3f} | "
            f"{raw_phase['single_support_ratio']:.3f}→{candidate_phase['single_support_ratio']:.3f}→{free_phase['single_support_ratio']:.3f} | "
            f"{source_event}→{candidate_event} | {value['gate']['pass']} |"
        )
    lines += [
        "",
        "## 裁决",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "- 该裁决只评价统一的 temporal collision-label schedule；不评价 X2 硬件接触力，也不否定闭环控制器或其他 reference 生成方法。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7", type=Path, default=DEFAULT_PHASE7)
    parser.add_argument("--phase7-report", type=Path, default=DEFAULT_PHASE7_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    phase7_motions = joblib.load(args.phase7)
    old_report = json.loads(args.phase7_report.read_text(encoding="utf-8"))
    role_to_key = {entry["panel_role"]: key for key, entry in phase7_motions.items()}
    phase2 = phase3.load_module(phase3.PHASE2_SCRIPT, "x2_phase2_helpers_for_phase9")
    physics = phase3.load_module(phase3.PHYSICS_SCRIPT, "x2_physics_helpers_for_phase9")

    per_role: dict[str, Any] = {}
    for role in ROLES:
        print(f"[phase9 trace-only] {role}", flush=True)
        key = role_to_key[role]
        entry = phase7_motions[key]
        frame_count = len(entry["dof"])
        old_role = old_report["per_role"][role]
        persisted = old_role["contact_comparison"]["per_side"]
        raw = {
            side: labels_from_events(frame_count, persisted[side]["official_events"])
            for side in ("left", "right")
        }
        source = {
            side: labels_from_events(frame_count, persisted[side]["fk_events"])
            for side in ("left", "right")
        }

        replay = phase4.simulate_contract_case(
            physics.DEFAULT_SCENE,
            physics.DEFAULT_CONTROL,
            entry,
            "free_root_balance",
            "zero",
            physics,
            phase2,
            expected_contact=raw,
            collect_contact_trace=True,
            contact_activation="collision",
        )
        old_free = old_role["physics"]["canonical"]["free"]
        reproduced = replay_reproduced(replay, old_free)
        if not reproduced["pass"]:
            raise RuntimeError(f"Phase7 free replay failed aggregate reproduction for {role}: {reproduced}")

        trace = replay.pop("contact_trace")
        times = np.asarray(trace["sim_times_s"], dtype=np.float64)
        keep = times <= replay["simulated_duration_s"] + 1.0e-12
        times = times[keep]
        common_frames = min(frame_count, int(np.floor(replay["simulated_duration_s"] * entry["fps"] + 1.0e-9)) + 1)
        active = {
            side: np.asarray(trace["active"][side], dtype=bool)[keep]
            for side in ("left", "right")
        }
        realized = phase4.contact_frame_labels(active, times, common_frames, float(entry["fps"]))
        raw_common = {side: raw[side][:common_frames] for side in ("left", "right")}
        source_common = {side: source[side][:common_frames] for side in ("left", "right")}
        candidate = canonical_schedule(raw_common)

        source_comparison = phase4.compare_contact_labels(source_common, realized)
        raw_comparison = phase4.compare_contact_labels(raw_common, realized)
        candidate_comparison = phase4.compare_contact_labels(candidate, realized)
        raw_phase = phase_statistics(raw_common)
        source_phase = phase_statistics(source_common)
        candidate_phase = phase_statistics(candidate)
        realized_phase = phase_statistics(realized)
        preservation = preservation_gate(raw_phase, candidate_phase)
        gate = schedule_gate(source_comparison, candidate_comparison, preservation)

        per_role[role] = {
            "motion_key": key,
            "common_survival_frames": common_frames,
            "common_survival_duration_s": float((common_frames - 1) / entry["fps"]),
            "phase7_aggregate_reproduction": reproduced,
            "replay_observation": {name: value for name, value in replay.items()},
            "label_sha256": {
                "raw_prescribed": array_sha256(raw_common),
                "phase7_source_intent": array_sha256(source_common),
                "canonical_schedule": array_sha256(candidate),
                "free_realized": array_sha256(realized),
            },
            "phase_statistics": {
                "raw_prescribed": raw_phase,
                "phase7_source_intent": source_phase,
                "canonical_schedule": candidate_phase,
                "free_realized": realized_phase,
            },
            "comparison_to_free_realized": {
                "raw_prescribed": raw_comparison,
                "phase7_source_intent": source_comparison,
                "canonical_schedule": candidate_comparison,
            },
            "event_error_summary": {
                "phase7_source_intent": event_error_summary(source_comparison),
                "canonical_schedule": event_error_summary(candidate_comparison),
            },
            "preservation_gate": preservation,
            "gate": gate,
        }

    reproduction_all = all(value["phase7_aggregate_reproduction"]["pass"] for value in per_role.values())
    improved_all = all(value["gate"]["pass"] for value in per_role.values())
    failed_roles = [role for role, value in per_role.items() if not value["gate"]["pass"]]
    status = "PHASE9_CONTACT_SCHEDULE_FROZEN" if improved_all else "PHASE9_CONTACT_SCHEDULE_REJECTED"
    report = {
        "schema_version": "x2_wbt_contact_schedule_phase9_v1",
        "provenance": {
            "phase7_bronze": {"path": str(args.phase7), "sha256": phase3.sha256(args.phase7)},
            "phase7_report": {"path": str(args.phase7_report), "sha256": phase3.sha256(args.phase7_report)},
            "official_scene": {"path": str(physics.DEFAULT_SCENE), "sha256": phase3.sha256(physics.DEFAULT_SCENE)},
            "official_control": {"path": str(physics.DEFAULT_CONTROL), "sha256": phase3.sha256(physics.DEFAULT_CONTROL)},
        },
        "truth_boundary": {
            "labels_are_official_simulator_collision_estimates": True,
            "not_hardware_grf_cop_wrench_or_foot_force": True,
            "one_trace_only_free_replay_per_motion": True,
            "second_physics_pass": False,
            "stochastic_seed": "none; frozen MuJoCo replay contains no stochastic sampling",
            "ik_stance_lock_training_teacher_checkpoint_base_git_baidu_real_robot": False,
        },
        "frozen_phase7_contract": {
            "motion_root_ground_joint_timing_unchanged": True,
            "formula": "z'=z+MA9(c_reset-min_24_active_sole_signed_distance)",
        },
        "pre_registered_schedule": {
            "input": "Phase7 canonical prescribed-root official collision event lists",
            "fps": 30,
            "on_confirm_frames": ON_CONFIRM_FRAMES,
            "off_confirm_frames": OFF_CONFIRM_FRAMES,
            "minimum_dwell_frames": MIN_DWELL_FRAMES,
            "clip_specific_parameters": False,
            "free_survival_used_to_select_parameters": False,
            "segmentation_states": ["double_support", "left_single_support", "right_single_support", "flight"],
        },
        "per_role": per_role,
        "decision": {
            "phase7_aggregate_reproduction_all_five": reproduction_all,
            "consistent_improvement_all_five": improved_all,
            "failed_roles": failed_roles,
            "status": status,
            "result": (
                "统一 contact-label schedule 跨五动作改善并保留单支撑语义。"
                if improved_all
                else f"统一 schedule 未在 {failed_roles} 相对 Phase7 source intent 一致改善；按预注册门停止。"
            ),
            "conclusion": (
                "可冻结为下一阶段的模型接触标签契约，但尚不是动态Silver或硬件接触真值。"
                if improved_all
                else "固定的hysteresis/min-dwell不能把prescribed collision事件统一对齐free-root realized contact；否定该temporal schedule，不修改Phase7 Bronze。"
            ),
            "next_step": (
                "仅把标签契约供下一阶段使用；不得把它解释为GRF/COP，也不在本阶段做IK。"
                if improved_all
                else "不扩hysteresis/dwell/clip参数；保留Phase7 Bronze，转向闭环contact-conditioned控制或具有连续相位的轨迹方法。"
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
