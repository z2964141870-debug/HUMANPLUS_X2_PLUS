# Privileged teacher role-dose v2e

Decision: `FAIL_NO_COMPLETE_ROLE_DOSE_WINDOWS_STOP`.

The run completed all 8 cyclic passes (`128 env × 512 steps` per pass), produced all 1,024 trajectory records and all 64 role-dose cells, replayed the reset fingerprints exactly, and left the source model unchanged. It used 0 optimizer steps and wrote 0 checkpoints.

Only `vx_0p20_b` had a safe nonzero cell, at scale `0.10`. The other seven roles had no dose that passed all preregistered posture and safety gates. The transition source cell itself was invalid because at least one trajectory did not reach the terminal window. Turn and 0.35 m/s cells were primarily blocked by support/root-height safety; 0.50 m/s cells traded pitch signal against slip/support; `vx_0p20_a` remained termination- and stability-limited.

This closes the fixed open-loop phase/contact teacher direction. No additional scale sweep, teacher dataset, BC/DAgger, PPO, export, or deployment is unlocked. The next admissible method is a state-feedback privileged teacher/WBC that can coordinate correction with velocity, support margin and transition state.

Evidence: result SHA-256 `5d79c9ccd368643ac03d96f384c52ea0f514ae1781605b7dc891a881906edd52`; resource SHA-256 `1ab59fb3d7346c44f96a6d823e95368468d27342bcdfc1a05796056e08d5f4b8`; log SHA-256 `dcf580e0cf58c0a2e42f2cdc2513dc5317c27cdaa62e33758c1bbaa1f62734b0`. Runtime was `227.03 s`, peak GPU memory `3271 MiB`, and disk delta `135,516,160 bytes`.
