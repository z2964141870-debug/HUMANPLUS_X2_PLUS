# Brake/handoff panel v5b — lifecycle-invalid diagnostic

The complete screen was written, but Isaac shutdown exceeded the 1200-second deadline. The supervisor sent SIGTERM, so the run is formally invalid and the independent finalizer was not run.

One result is still worth carrying forward only as a hypothesis: keeping the locomotion actor active under a zero command (`locomotion_zero`) completed hold in 16/16 screening lanes with speed p95 0.061 m/s, mean pitch −0.004 rad, full double support, and no termination. In the 256-lane replay, 250 lanes reached and retained hold; six failed earlier during cruise. The direct locomotion-to-stationary mix had 218 hold terminations among the same-size replay.

This is not a pass and cannot unlock deployment or training. It only justifies a fresh-seed, no-training confirmation of `locomotion_zero`, without changing thresholds based on these results.
