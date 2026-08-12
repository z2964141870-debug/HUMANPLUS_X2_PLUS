# X2 privileged-generator Phase40 new-machine live-zero — 2026-08-12

## Decision

`PASS_LIVE_ZERO_ONLY`

The migrated machine completed one real Isaac/PhysX reset from the native-generator seed. The smoke stayed inside its zero-update boundary: no environment control step, reward evaluation, PPO update, optimizer instance, optimizer step, or checkpoint creation occurred. This result validates the Phase40 live wiring only; it does not by itself unlock training.

Machine-readable result: `reports/retarget/x2_privileged_generator_live_zero_new_machine_20260812.json`

## Verified runtime contract

- Environment instances: 1
- Environment resets: 1
- Environment control steps: 0
- Optimizer instances / steps: 0 / 0
- Checkpoints created: 0
- WBT action dimension: 29
- Future observation shape: 10 x 58
- All observations finite: yes
- No contact labels written by the native reset: yes
- Post-manager finalizer applied: yes
- Maximum PhysX writeback error: `1.7881393432617188e-07` (root pose); all other measured PhysX/cache errors were zero
- Hard joint-position projection applied: no
- Maximum soft-limit overshoot reported by the seed: `0.1453222781419754 rad`

Result JSON SHA256: `af4b230c976771952bcb1b42b0e6714c6f125898bd1478d0d2d40610c9551ae7`

## Restored Stage219 provenance

The Baidu download archive `/home/yu/projects/data/【批量下载】env.yaml等.zip` passed ZIP integrity validation and supplied the two previously missing Stage219 parameter files. They were restored without overwriting an existing target:

- `env.yaml`: `f702a358bdbc1df94ac2a54b83aa4f6d7c98c76b091ac05a65fd074c44e6f9d7`
- `agent.yaml`: `38d462ad726e0e74d797f8a0ce3799aaadc02737e6e14a5cdf3443f7da0a8368`

Both hashes exactly match the roadmap's expected values. The downloaded SONIC checkpoint at `/home/yu/projects/data/last.pt` was also reverified as `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.

## Reproduction

```bash
PYTHONPATH=.:src conda run -n x2-sonic-isaaclab \
  pytest -q tests/test_phase40_privileged_generator_live_wiring.py

PHASE40_OUTPUT="$PWD/reports/retarget/x2_privileged_generator_live_zero_new_machine_20260812.json" \
  bash scripts/run_phase40_privileged_generator_live_zero.sh
```

Observed results: wiring tests `4 passed`; launcher exit status `0`; explicit report postcondition verified `PASS_LIVE_ZERO_ONLY`, `optimizer_steps == 0`, and `checkpoints_created == 0`.

The pre-existing 2026-08-11 failed report at `research/dynamic_retargeting_20260811/phase40_privileged_generator_live_zero.json` was intentionally left unchanged for audit history.

## Storage note

After dependency repair and the live run, `/dev/nvme0n1p2` has 352 GiB available (21% used). The Baidu ZIP is 44 KiB, the first-run Isaac extension cache is 191 MiB, and the complete `x2-sonic-isaaclab` environment occupies 21 GiB. No checkpoint was produced by this smoke.
