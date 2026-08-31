# X2 Sonic Agent Rules

These rules apply to this repository. They are intentionally stricter than
the historical launch scripts because this checkout controls a real X2.

## Read first

Before changing code, deploying, or connecting a powered controller, read in
this order:

1. `docs/CURRENT_STATE.md`
2. `docs/DECISIONS.md`
3. `docs/OPERATIONS.md`
4. `docs/EXPERIMENTS.md`
5. `gear_sonic_deploy/README_X2_GARMENT_LIVE.md`
6. `docs/BASELINE_170450.md`
7. `gear_sonic_deploy/README_STAGE_SONIC_BALANCE.md`

The files above are the handoff state. Chat history is supporting evidence,
not the operational source of truth.

## Current stop condition

As of 2026-08-31, the robot is fully suspended and offline after `lifted`
normally ended the custom writer. The resumed SoC0 MC worker then exited and
all four HAL command topics were observed with zero publishers before network
loss. Official MC ownership is therefore unverified. Do not lower the robot,
start a writer, or assume MC recovered until connectivity returns and the
documented MC restart plus one-official-publisher checks pass.

The next bounded standing experiment is D-017: establish the final foot
contact and sling tension while policy is off, capture that state with `load`,
then keep feet and sling unchanged for one 100-second fixed-support run. Do not
repeat an on-policy sling adjustment or use absolute waist error as an entry
target.

## Frozen architecture

- Robot firmware: `test-lx2501_3_t2d5-soc1-dc-v0.9.0-rc7`.
- Message ABI: `aimdk_msgs 0.8.18`.
- Sonic host: SoC1. HAL and official MC host: SoC0.
- Control path: direct HAL at a 250 Hz writer rate; policy inference at 50 Hz.
- Policy preset: `neutral_damped` with reconstructed pelvis IMU.
- The next powered candidate is the hash-gated
  `gear_sonic_deploy/run_x2_supported_preloaded_fixed100.sh`; it remains
  unauthorized until MC ownership recovery, remote hash verification, and a
  fresh physical safety confirmation.
- `run_x2_ground_load_hold.sh` is allowed only for policy-disabled static-PD
  diagnostics.
- `run_x2_suspended_sonic.sh` is an internal implementation entry and must not
  be launched directly.
- `run_x2_supported_neutral_damped_support_step.sh` and every other
  `run_x2_*.sh` wrapper are frozen experimental history. Do not run one unless
  `docs/DECISIONS.md` and `docs/EXPERIMENTS.md` are first updated with a
  single-variable rationale and explicit user authorization.

Do not reopen v1.0 `Develop_MC`, 500 Hz publishing, `waist2`, `waist2-anklemc`,
`/aima/mc/joint/retargeting`, or gain tuning while the supported neutral entry
gate is unresolved. Offline integration of the onboard garment/ZMQ path is now
active and must follow `README_X2_GARMENT_LIVE.md`. It is not authorization to
put garment input into the powered HAL loop before that README's acceptance
gates close.

## Real-hardware rules

Before any command that can publish custom HAL commands, obtain a current
operator confirmation that the robot is powered, fully suspended, and a
person is holding the physical emergency stop. Then perform the read-only
preflight in `docs/OPERATIONS.md`.

Never send `policy` merely because the launcher reached `READY_FOR_GROUND`.
Record the static entry state first. The known-good posture window in the
operations guide is a reproduction target derived from one run, not a general
safety theorem.

While partly loaded, do not use `Ctrl-C`, close the SSH session, or type
`lifted`. Use `stop` to return from policy to static PD. Re-tension and fully
suspend the robot before typing `lifted`, which ends the writer and restores
official MC. The physical emergency stop overrides all software procedures.

Use `/usr/bin/python3` for ROS Python probes. The base Conda Python is 3.13 and
is incompatible with the ROS Humble Python 3.10 packages on this robot.

## Experiment discipline

- Change one primary variable per powered experiment.
- Keep the model, gains, filters, 250 Hz writer, pelvis reconstruction, safety
  limits, and launcher fixed during the entry-posture experiment.
- Give every run a hypothesis, exact command, preconditions, success/failure
  criteria, log path, hashes, result, and recovery state in
  `docs/EXPERIMENTS.md`.
- Preserve failed logs. Do not delete or rewrite raw CSVs.
- Say exactly what evidence proves. A suspended or gantry-supported result
  does not prove unsupported balance, walking, or garment teleoperation.
- Correlation between entry posture and outcome is not a root-cause proof.

## Repository hygiene

The local `main` checkout is intentionally dirty and contains substantial
user work. Do not reset, clean, reformat, rename, or overwrite unrelated
modified/untracked files. Do not initialize Git in the SoC1 deployment copy;
it is a synchronized runtime tree, not a repository.

Do not record passwords, tokens, cookies, or private keys in scripts, docs,
logs, or shell history. Do not commit or push unless the user explicitly asks.
Before copying to SoC1, compare the exact source and destination paths and
SHA-256 hashes.
