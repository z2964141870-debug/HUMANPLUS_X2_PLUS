# Phase75 seed0 resource failure freeze

- Decision: `FAIL_TECHNICAL_STOP`; the launch is not accepted.
- The initial-only technical screen itself passed all 268 field diagnostics for 64 pairs and wrote a commit and screen with valid sidecars.
- The resource ledger recorded 144.31169396899713 seconds, exceeding the preregistered 120-second hard wall limit by 24.311693968997133 seconds. All other recorded resource gates passed.
- The child ledger records exit code 0 and the immutable child log has no traceback. The externally observed root SIGTERM at approximately 136 seconds is retained as orchestration provenance, not as a claim derived from the child log.
- The outer shell failed the wall-time resource gate; seed1 and seed2 were not started. There is no Phase75 final result.
- No scored physics rollout, scientific metric, gradient, optimizer step, or checkpoint was produced. Phase76 launch, training, and deployment remain locked.

The machine-readable freeze JSON and its SHA256 sidecar are authoritative. The original preregistration, commit, screen, resource ledger, log, immutable code and immutable inputs are retained in the independent recovery inventory.
