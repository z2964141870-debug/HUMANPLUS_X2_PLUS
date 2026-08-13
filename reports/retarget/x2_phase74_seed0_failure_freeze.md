# Phase74 seed0 failure freeze

- Decision: `FAIL_TECHNICAL_STOP`.
- Scientific result: none; this was an initial-only technical preflight.
- Seed0 stopped before its commit and screen were written. Seed1 and seed2 were not started.
- The observed traceback is `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)` at the `tensor_sides` donor-hash indexing operation.
- The GPU ledger correctly records exit code 1, 13.531275330002245 seconds elapsed, 3,197 MiB peak GPU memory, and 143,065,088 bytes of disk growth.
- No physics rollout, gradient, optimizer step, or checkpoint was authorized or produced. Phase75 launch, training, and deployment remain locked.

The authoritative provenance is the machine-readable freeze JSON and its SHA256 sidecar. The original preregistration, failure JSON, resource ledger, log, immutable code, and immutable inputs are retained in the independent backup inventory.
