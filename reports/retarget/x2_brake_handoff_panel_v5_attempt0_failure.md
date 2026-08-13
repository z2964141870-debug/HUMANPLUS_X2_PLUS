# Brake/handoff panel v5 attempt 0

Attempt 0 stopped before environment construction because the shell omitted the `sonic_x2_sandbox` import path. No physics rollout, controller comparison, screen, optimizer, backward pass, or checkpoint occurred. The ledger's raw return code is not authoritative because the exception was outside the runner's guarded `run()` block; the immutable traceback is authoritative.

A separately frozen v5b may repair only the shell environment and use fresh output paths. It must not change the strategies, seeds, thresholds, or decision rules.
