#!/usr/bin/env python3
"""Clean Phase59 checkpoints and write the final compact audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    source_remains_best = state["best_candidate"] is None
    if source_remains_best:
        first_gate = json.loads(Path(state["completed_updates"][0]["gate"]).read_text())
        state["best_update"] = 0
        state["best_joint_score"] = float(first_gate["scores"]["A_pre"] + first_gate["scores"]["B_pre"])
        best_final = args.source
    else:
        best_source = Path(state["best_candidate"])
        best_final = args.artifact_dir / f"best_update{state['best_update']:02d}.pt"
        if best_source != best_final:
            best_source.replace(best_final)
    rejected_final = None
    if state["rejected_candidate"] is not None:
        rejected_source = Path(state["rejected_candidate"])
        rejected_final = args.artifact_dir / f"rejected_last_update{state['rejected_update']:02d}.pt"
        if rejected_source != rejected_final:
            rejected_source.replace(rejected_final)
    keep = {args.source.resolve(), best_final.resolve()}
    if rejected_final is not None:
        keep.add(rejected_final.resolve())
    removed = []
    for path in args.artifact_dir.glob("*.pt"):
        if path.resolve() not in keep:
            removed.append(path.name)
            path.unlink()
    if source_remains_best:
        decision = "EARLY_STOP_UPDATE1_NO_PASS_SOURCE_REMAINS_BEST"
    else:
        decision = "EARLY_STOP_AFTER_REJECT_KEEP_BEST" if rejected_final else "FIVE_UPDATES_PASS_KEEP_BEST"
    report = {
        "phase": 59, "decision": decision,
        "updates_attempted": len(state["completed_updates"]),
        "completed_updates": state["completed_updates"],
        "best_update": state["best_update"], "best_joint_score": state["best_joint_score"],
        "weights": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "best": {"path": str(best_final), "sha256": sha256(best_final), "same_as_source": source_remains_best},
            "rejected_last": None if rejected_final is None else {"path": str(rejected_final), "sha256": sha256(rejected_final)},
        },
        "removed_redundant_checkpoints": sorted(removed),
        "retained_checkpoint_count": len(keep),
        "boundary": "At most five updates; best selected by joint A+B score; no automatic longer training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
