#!/usr/bin/env python3
"""Update Phase59 early-stop/best state from one preregistered gate result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    if gate["phase"] != 59:
        raise RuntimeError("selector received a non-Phase59 gate")
    state = json.loads(args.state.read_text()) if args.state.exists() else {
        "phase": 59, "completed_updates": [], "best_update": None,
        "best_joint_score": None, "best_candidate": None,
        "rejected_update": None, "rejected_candidate": None,
    }
    passed = gate["decision"] == "PASS_LORA_ONE_UPDATE_TREND_GATE_STOP"
    joint_score = float(gate["scores"]["A_post"] + gate["scores"]["B_post"])
    state["completed_updates"].append({
        "update": args.update, "passed": passed, "joint_score": joint_score,
        "gate": str(args.gate), "candidate": str(args.candidate),
    })
    if passed:
        if state["best_joint_score"] is None or joint_score > state["best_joint_score"]:
            state["best_update"] = args.update
            state["best_joint_score"] = joint_score
            state["best_candidate"] = str(args.candidate)
    else:
        state["rejected_update"] = args.update
        state["rejected_candidate"] = str(args.candidate)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"passed": passed, "joint_score": joint_score, "best_update": state["best_update"]}))
    raise SystemExit(0 if passed else 10)


if __name__ == "__main__":
    main()
