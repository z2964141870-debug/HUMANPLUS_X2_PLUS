# X2 dynamic-retargeting execution state

Updated: 2026-08-11
Scope: companion state file for `X2_DYNAMIC_RETARGETING_MULTI_METHOD_AGENT_CARD_20260810.md`

## Reviewer authorization

`PHASE1_UPSTREAM_SMOKE_AUTHORIZED = true`

The reviewer explicitly authorizes the executing agent to:

1. use network access to create a new isolated Conda environment, recommended name `x2-dsms-agent-b`;
2. clone/install the pinned upstream Shooting for Contact dependencies;
3. run the upstream `cartpole` smoke;
4. if cartpole passes, run the upstream `g1_gait` smoke;
5. monitor long-running work at approximately 30-minute intervals.

Creating an isolated Conda environment is treated as reversible and already
authorized. It is not a reason to enter a 1500-second reviewer wait. If a tool
requires a platform permission prompt, request that permission immediately.

## Overnight autonomous loop authorization

The user's latest instruction authorizes an unattended working loop while the
user sleeps:

- `PHASE1_UPSTREAM_SMOKE_AUTHORIZED = true`
- `PHASE2_X2_CONTRACT_AUTHORIZED_IF_PHASE1_PASSES = true`
- `PHASE3_X2_LUNGE_NLP_AUTHORIZED_IF_PHASE2_PASSES = true`
- `PHASE4_RL_TRAINING_AUTHORIZED = false`

The agent must self-evaluate the preregistered gates. If a phase passes, proceed
to the next authorized phase without waiting for another reviewer message. If
a hard gate fails, stop that route, preserve the failure evidence and follow
the task card's next-method/blocked reporting rule. Do not weaken a gate to
keep the loop alive.

During an active long solve, check progress approximately every 30 minutes.
A long fallback heartbeat is acceptable only while a real process is running
or recovering from an external transient; it must not replace immediately
available work.

Stop before any RL/PPO/DC-PEFT optimizer run. Also stop for credential prompts,
destructive operations, license ambiguity that forbids internal execution, or
any requested HSL/MA57 download.

HSL/MA57 download or redistribution is not authorized. Use legally available
MUMPS unless a separate solver authorization is given.

## Corrected Phase0 evidence reported by Agent B

The earlier statement that forward4 had approximately 33 mm of true foot
penetration was over-stated. After applying the Phase7 canonical root-ground
convention, Agent B reports:

- the apparent penetration was dominated by a root-z convention offset;
- the required mean root-z correction was approximately 11--18 mm;
- residual penetration was approximately 0.3--0.6 mm;
- contact-timing contradiction remained approximately 63--71%;
- foot slip remained approximately 0.88--1.04 m/s;
- self-collision remained approximately 20--29%;
- root horizontal acceleration remained approximately 14--15 m/s^2.

Therefore Phase0 may still justify dynamic-feasibility work, but its main
evidence is contact contradiction, slip, self-collision and root acceleration,
not the old 33 mm penetration claim. Preserve the raw and regrounded audits and
their exact artifact hashes in the Phase0 report.

## Immediate next action

Do not re-arm an idle reviewer monitor. Start Phase1 now:

```text
create isolated env
-> freeze repository and dependency versions
-> license audit
-> cartpole smoke
-> g1_gait smoke if cartpole passes
-> self-check Phase1 gates
-> if PASS, build and test Phase2 X2 zero-contract
-> if PASS, run the preregistered Phase3 short-prefix lunge NLP
-> expand to full lunge only if the short-prefix gate passes
-> stop before RL training and submit the complete overnight report
```
