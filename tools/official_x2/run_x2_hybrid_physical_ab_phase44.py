#!/usr/bin/env python3
"""Default-fail-closed Phase44 hybrid physical A/B runner.

The default invocation only validates schemas/hashes and writes the immutable
upper-motion/control contract. ``--execute`` is rejected unless BASE Phase28
explicitly qualifies a native dynamic seed and supplies the required extended
contact instrumentation contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_panel_phase28 as phase28
import retarget.audit_x2_hybrid_composition_contract_phase43 as phase43


BASE_PHASE28 = REPO / "reports/official_x2/phase28_stage250_extended_direct_manifest.json"
PHASE43_REPORT = phase43.OUTPUT_JSON
PHASE43_ARTIFACT = phase43.OUTPUT_NPZ
UPPER_NPZ = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
CONTRACT_JSON = REPO / "reports/retarget/x2_hybrid_physical_ab_phase44.json"
CONTRACT_MD = REPO / "reports/retarget/x2_hybrid_physical_ab_phase44.md"
GATE_RUNNER = REPO / "tools/official_x2/run_official_gate_case.sh"

UPPER_SCALE = 0.25
UPPER_MAX_EXCURSION_RAD = 0.12
UPPER_MAX_VELOCITY_RADPS = 0.20
UPPER_TIME_SCALE = 1.0
CONTROL_DT = 0.02
UPPER_PD = {
    "shoulder_elbow": {"kp": 40.0, "kd": 5.0},
    "wrist": {"kp": 30.0, "kd": 3.0},
}


def phase28_qualification(manifest: dict[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    if manifest is None:
        return False, {"reason": "manifest_missing"}
    decision = manifest.get("decision", {})
    if isinstance(decision, bool):
        qualified = decision
    elif isinstance(decision, str):
        qualified = decision.lower() in ("qualified", "pass", "passed", "ready")
    else:
        qualified = any(decision.get(key) is True for key in (
            "qualified", "native_dynamic_seed_qualified", "usable_as_contact_consistent_dynamic_seed",
        ))
    # Physical A/B additionally requires explicit substep realized-contact
    # instrumentation; a generic qualified flag alone cannot silently unlock.
    text = json.dumps(manifest, ensure_ascii=False).lower()
    instrumentation = all(token in text for token in ("contact", "substep")) and any(
        token in text for token in ("impulse", "force")
    )
    return bool(qualified and instrumentation), {
        "manifest_decision": decision, "qualified_flag": bool(qualified),
        "extended_contact_instrumentation_declared": bool(instrumentation),
    }


def bounded_upper_target(raw_upper: np.ndarray, default: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = raw_upper[0].copy()
    residual = UPPER_SCALE * (raw_upper - baseline)
    residual = np.clip(residual, -UPPER_MAX_EXCURSION_RAD, UPPER_MAX_EXCURSION_RAD)
    desired = default[None] + residual
    bounded = np.empty_like(desired)
    previous = default.copy()
    max_step = UPPER_MAX_VELOCITY_RADPS * CONTROL_DT
    for frame in range(len(desired)):
        previous = previous + np.clip(desired[frame] - previous, -max_step, max_step)
        bounded[frame] = previous
    return bounded, residual


def stage250_env(case: str, domain: int, result_root: Path) -> dict[str, str]:
    return {
        "CASE_NAME": case, "ROS_DOMAIN_ID": str(domain), "RESULT_ROOT": str(result_root),
        "MODEL_PATH": "/models/stage219_s2600_actor.onnx", "COMMAND_VX": "0.30",
        "STATE_QOS_DEPTH": "1", "STATE_PREDICTION_SECONDS": "0",
        "MOVE_SECONDS": "4.0", "STOP_SECONDS": "8.0", "STOP_CONTROLLER": "policy",
        "HEADING_GAIN": "0.50", "HEADING_RATE_LIMIT": "0.10",
        "ACTION_BIAS_MODE": "lateral_recovery_supervisor", "ACTION_BIAS": "0.50",
        "ANKLE_ROLL_COMMON_BIAS": "0.20", "RECOVERY_ENTER_M": "0.12",
        "RECOVERY_EXIT_M": "0.04", "RECOVERY_SLEW_RATE_PER_S": "1.0",
        "MAX_ATTEMPTS": "1", "TIMEOUT_SECONDS": "180",
    }


def run_case(env: dict[str, str]) -> Path:
    subprocess.run(["bash", str(GATE_RUNNER)], env={**os.environ, **env}, check=False)
    result = Path(env["RESULT_ROOT"]) / f"{env['CASE_NAME']}.json"
    if not result.exists():
        raise RuntimeError(f"official runner produced no result: {result}")
    return result


def render(report: dict[str, Any]) -> str:
    q = report["qualification"]
    return f"""# X2 WBT Phase44：hybrid physical A/B runner合同

## 状态

**{report['decision']['status']}**

- BASE Phase28 manifest exists：`{q['manifest_exists']}`；qualified+instrumented：`{q['execution_unlock']}`。
- 本次physics执行：`{report['truth_boundary']['physics_executed']}`；PPO/optimizer：`False`。

## A/B

- A：Phase28/Stage250 native straight，lower/root/waist/head原生。
- B：完全相同native backend + `AMASS-UPPER-001` upper14；GMR lower/root/contact禁止进入。
- upper raw 30→50Hz由Phase43冻结；official adapter合同：`default + clip(0.25*(q-q0), ±0.12rad)`，再以0.20rad/s限速，输出绝对PD target。
- shoulder/elbow PD=40/5，wrist PD=30/3；official joint order固定。

## 静态结果

- raw/bounded upper shapes：{report['upper_contract']['raw_shape']} / {report['upper_contract']['bounded_shape']}。
- bounded target qstep max：{report['upper_contract']['bounded_qstep_max_rad']:.6f}rad（门={UPPER_MAX_VELOCITY_RADPS*CONTROL_DT:.6f}）。
- lower/waist/root/head composition round-trip：`{report['static_checks']['native_owned_fields_exact']}`。

## 门禁

- A必须先完全复现qualified seed；A失败不运行B。
- prescribed/free严格分栏；prescribed不得表述为free balance。
- B报告upper RMSE/qstep、survival、realized contact、slip、signed pitch；composer不得直接改lower target。

## 结论

{report['decision']['conclusion']}

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--domain-base", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=REPO / "artifacts/official_x2/phase44_results")
    args = parser.parse_args()
    if not PHASE43_REPORT.exists() or not PHASE43_ARTIFACT.exists():
        raise FileNotFoundError("Phase43 contract/artifact required")
    phase43_report = json.loads(PHASE43_REPORT.read_text())
    dry = np.load(PHASE43_ARTIFACT)
    official31 = tuple(dry["official31"].tolist())
    upper_indices = np.asarray([official31.index(name) for name in phase43.UPPER14], dtype=np.int64)
    raw_upper = np.asarray(dry["composite_q31"], dtype=np.float64)[:, upper_indices]
    model_contract = json.loads(phase43.MODEL_CONTRACT.read_text())
    joint_rows = {row["name"]: row for row in model_contract["joints"]}
    default = np.asarray([joint_rows[name]["model_nominal_rad"] for name in phase43.UPPER14])
    bounded, residual = bounded_upper_target(raw_upper, default)
    limits = np.asarray([joint_rows[name]["range_rad"] for name in phase43.UPPER14])
    limit_overshoot = np.maximum(limits[:, 0][None] - bounded, 0.0) + np.maximum(bounded - limits[:, 1][None], 0.0)
    UPPER_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        UPPER_NPZ, joint_names=np.asarray(phase43.UPPER14), q_rad=raw_upper.astype(np.float32),
        fps=np.asarray(50.0), source=np.asarray("Phase43 AMASS-UPPER-001 first4s real-time 30to50"),
        prereg_bounded_absolute_target_rad=bounded.astype(np.float32),
        prereg_residual_rad=residual.astype(np.float32),
        kp=np.asarray([40.0 if "wrist" not in name else 30.0 for name in phase43.UPPER14]),
        kd=np.asarray([5.0 if "wrist" not in name else 3.0 for name in phase43.UPPER14]),
    )
    manifest_exists = BASE_PHASE28.exists()
    manifest = json.loads(BASE_PHASE28.read_text()) if manifest_exists else None
    unlocked, qualification_detail = phase28_qualification(manifest)
    static_checks = {
        "phase43_hash_frozen": phase28.sha256(PHASE43_ARTIFACT) == phase43_report["artifact"]["sha256"],
        "official_joint_order_exact": official31 == tuple(model_contract["control_boundaries"]["official_mjcf_actuated_31"]),
        "upper_joint_order_exact": tuple(phase43.UPPER14) == tuple(UPPER_NPZ and np.load(UPPER_NPZ)["joint_names"].tolist()),
        "bounded_rate": float(np.max(np.abs(np.diff(bounded, axis=0)))) <= UPPER_MAX_VELOCITY_RADPS * CONTROL_DT + 1e-12,
        "bounded_excursion": float(np.max(np.abs(residual))) <= UPPER_MAX_EXCURSION_RAD + 1e-12,
        "bounded_joint_limits": float(np.max(limit_overshoot)) <= 1e-12,
        "native_owned_fields_exact": bool(phase43_report["roundtrip"]["B_native_fields_exact"] and phase43_report["roundtrip"]["B_root_exact"]),
        "gmr_lower_root_contact_absent": all(name in phase43_report["contract"]["forbidden"] for name in ("GMR lower", "GMR root", "GMR contact")),
    }
    static_pass = all(static_checks.values())
    executions: list[dict[str, Any]] = []
    physics_executed = False
    if args.execute:
        if not unlocked:
            raise SystemExit("BLOCKED_BY_NATIVE_SEED: BASE Phase28 is absent, unqualified, or lacks extended substep contact instrumentation")
        if not static_pass:
            raise SystemExit("BLOCKED_BY_STATIC_CONTRACT")
        if args.domain_base <= 0:
            raise SystemExit("--domain-base must be explicitly positive for execution")
        args.result_root.mkdir(parents=True, exist_ok=True)
        env_a = stage250_env("phase44_hybrid_A_native", args.domain_base, args.result_root)
        a_path = run_case(env_a); physics_executed = True
        a = json.loads(a_path.read_text())
        executions.append({"arm": "A", "path": str(a_path), "sha256": phase28.sha256(a_path), "summary": a.get("summary")})
        # A must reproduce the qualified seed before B is allowed.
        if not a.get("summary", {}).get("full_gate_pass"):
            raise SystemExit("A_DID_NOT_REPRODUCE_QUALIFIED_NATIVE_SEED")
        env_b = stage250_env("phase44_hybrid_B_upper", args.domain_base + 1, args.result_root)
        env_b.update({
            "UPPER_MOTION": "/repo/artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz",
            "UPPER_SCALE": str(UPPER_SCALE), "UPPER_TIME_SCALE": str(UPPER_TIME_SCALE),
            "UPPER_MAX_EXCURSION_RAD": str(UPPER_MAX_EXCURSION_RAD),
            "UPPER_MAX_VELOCITY_RADPS": str(UPPER_MAX_VELOCITY_RADPS),
        })
        b_path = run_case(env_b)
        b = json.loads(b_path.read_text())
        executions.append({"arm": "B", "path": str(b_path), "sha256": phase28.sha256(b_path), "summary": b.get("summary")})
    status = "PHASE44_PREFLIGHT_READY_EXECUTION_NOT_REQUESTED" if unlocked and static_pass else "BLOCKED_BY_NATIVE_SEED"
    conclusion = (
        "runner/schema/hash/upper-target合同通过，BASE Phase28也具备显式动态seed+substep contact资格；本次未请求执行。"
        if status.startswith("PHASE44_PREFLIGHT_READY")
        else "runner/schema/hash/upper-target合同已实现，但BASE Phase28尚未提供qualified且带substep realized-contact的native dynamic seed；默认fail-closed，未运行physics。"
    )
    report = {
        "schema_version": "x2_hybrid_physical_ab_phase44_v1",
        "truth_boundary": {"physics_executed": physics_executed, "ppo_or_optimizer": False, "prescribed_is_not_free_balance": True, "base_files_modified": False},
        "qualification": {"manifest": str(BASE_PHASE28), "manifest_exists": manifest_exists, "manifest_sha256": phase28.sha256(BASE_PHASE28) if manifest_exists else None, "execution_unlock": unlocked, **qualification_detail},
        "provenance": {"phase43_report_sha256": phase28.sha256(PHASE43_REPORT), "phase43_artifact_sha256": phase28.sha256(PHASE43_ARTIFACT), "gate_runner_sha256": phase28.sha256(GATE_RUNNER)},
        "upper_contract": {
            "representation": "absolute PD target = stage208 default + bounded residual from AMASS baseline",
            "raw_shape": list(raw_upper.shape), "bounded_shape": list(bounded.shape),
            "scale": UPPER_SCALE, "time_scale": UPPER_TIME_SCALE,
            "max_excursion_rad": UPPER_MAX_EXCURSION_RAD, "max_velocity_radps": UPPER_MAX_VELOCITY_RADPS,
            "bounded_qstep_max_rad": float(np.max(np.abs(np.diff(bounded, axis=0)))),
            "PD": UPPER_PD, "artifact": {"path": str(UPPER_NPZ), "sha256": phase28.sha256(UPPER_NPZ)},
        },
        "static_checks": static_checks,
        "ab_contract": {
            "A": "qualified BASE Phase28 Stage250 native straight lower/root/waist/head",
            "B": "same backend and commands plus Phase44 bounded AMASS upper14",
            "composition_direct_write_mask": list(phase43.UPPER14),
            "lower_direct_write_forbidden": list(phase43.LOWER12 + phase43.WAIST3),
            "A_must_reproduce_seed_before_B": True,
        },
        "gates": phase43_report["future_gate_contract"],
        "executions": executions,
        "decision": {
            "status": status, "physical_ab_executed": physics_executed,
            "conclusion": conclusion,
            "next_step": "等待BASE Phase28 manifest；若qualified则另行显式--execute，先A复现再B。" if not unlocked else "由主线决定是否显式执行；不得自动启动。",
        },
    }
    CONTRACT_JSON.write_text(json.dumps(phase28.json_safe(report), indent=2, ensure_ascii=False) + "\n")
    CONTRACT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
