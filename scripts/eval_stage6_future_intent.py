#!/usr/bin/env python3
"""Run the legacy Stage172 evaluator with the local Stage6 policy contract."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_SCRIPT = Path(
    os.environ.get(
        "CWI_STAGE6_BASE_EVAL",
        "/home/humanplus/x2_teleop_final/x2_sonic/scripts/"
        "eval_x2_stage172_lower_velocity.py",
    )
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("_cwi_stage6_base_eval", OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy evaluator: {OLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from cwi_x2.future_intent import (
        X2FutureIntentActorCriticCfg,
        X2FutureIntentFlatEnvCfg_PLAY,
    )
    from cwi_x2.future_intent_actor_critic import FutureIntentActorCritic

    mode = os.environ.get("CWI_STAGE6_ADAPTER_MODE", "future")

    def policy_cfg_factory():
        cfg = X2FutureIntentActorCriticCfg()
        # The legacy response-profile branch registers this exact class name.
        cfg.class_name = "ResponseHistoryActorCritic"
        cfg.adapter_mode = mode
        return cfg

    # Reuse the response-profile branch because it already records
    # base_actor_action and policy-minus-base residual in every trace.
    module.X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg_PLAY = (
        X2FutureIntentFlatEnvCfg_PLAY
    )
    module.X2ResponseHistoryActorCriticCfg = policy_cfg_factory
    module.ResponseHistoryActorCritic = FutureIntentActorCritic
    try:
        module.main()
    finally:
        module.simulation_app.close()


if __name__ == "__main__":
    main()
