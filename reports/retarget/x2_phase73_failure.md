# Phase73 failure freeze

- Decision: `FAIL_INVALID_STOP`.
- Scientific result: none. No Phase73 scientific direction was examined for this failure record.
- Seed0 plus completed one 400-step rollout and wrote its immutable raw bundle and technical screen.
- Seed0 minus started, then stopped before rollout with `RuntimeError: Phase73 pair initial tensors are not bitwise identical` while verifying the frozen pair initial state.
- Seed0 minus therefore has resource/log evidence but no raw bundle or screen. No pair result or final result exists.
- The seed0 minus resource ledger records exit code 0 despite the traceback. That exit code is not reliable; the shell stopped because the required screen was absent.
- Two launches were started, one rollout completed, one launch failed before rollout, and eight launches were never started. Replacement and continuation are forbidden by the preregistered stop rule.
- Optimizer/checkpoint authority remained locked; the completed plus screen records zero optimizer steps, backward calls, and checkpoints.

The authoritative machine-readable provenance is `x2_phase73_failure.json` and its SHA256 sidecar. Raw values are preserved solely for recovery and were not interpreted here.
