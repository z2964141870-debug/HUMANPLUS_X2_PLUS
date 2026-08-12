# X2 native locomotion Task2 Phase60 — preregistered A/B/C local pilot

## Decision

`FAIL_BOTH_LOCAL_TREND_STOP`

Neither trained arm is promoted. Both candidates remain failed experimental evidence and must not be used as a deployment backend or continued as though they passed. Training remains locked beyond this one-update pilot.

## Contract

- A: frozen Stage219/Stage264 identity, original reward, no optimizer.
- B: fresh Stage219 plus a one-sided backward-pitch penalty.
- C: fresh Stage219 plus the same pitch penalty and a live-mass COM penalty relative to actual ground-contact support.
- B/C each trained one rank-4 zero-output LoRA update: 64 env × 24 steps = 1536 transitions; 5 epochs × 4 minibatches = 20 optimizer steps; fixed LR `5e-5`.
- Dense actor/critic and action std stayed byte-stable. Upper-body targets were fixed in all 64 environments. No direct waist-pitch limit and no always-upright constraint was added.
- One shared source and one final checkpoint per trained arm were retained; no intermediate checkpoint was written.

The preregistration SHA256 was `5a9806d0c658ed6e2aedb6085c90dd4a4554d9dba0ddf4fe200a9383a7811035` before either optimizer ran.

## Results

| metric | A frozen | B pitch-only | C pitch+support |
|---|---:|---:|---:|
| signed pitch mean | -0.193130 rad | -0.191290 rad | -0.191077 rad |
| improvement vs A | — | +0.001840 rad | +0.002053 rad |
| signed pitch p05 | -0.246841 rad | -0.246979 rad | -0.247336 rad |
| COM-support outside mean | 0.061384 m | 0.061324 m | 0.061585 m |
| velocity tracking RMSE | 0.105676 m/s | 0.107438 m/s | 0.107603 m/s |
| stance slip p95 | 0.286816 m/s | 0.266429 m/s | 0.252421 m/s |
| survival / termination | 4.0 s / 0 | 4.0 s / 0 | 4.0 s / 0 |
| fixed-observation KL mean | — | 0.00005261 | 0.00005172 |
| action drift max | — | 0.001149 | 0.001156 |

B failed the preregistered mean-pitch improvement (`0.001840 < 0.002 rad`) and p05 non-regression gates. C cleared the mean-pitch threshold by only `0.000053 rad`, but failed both p05 non-regression and the required COM-support non-regression. Speed, survival, action rate, root height, tilt, knee excursion, swing clearance, flight fraction, and stance-slip limits passed for both.

The three evaluation seeds produced numerically identical physical metrics because observation corruption was disabled and reset pose/command were fixed. They are reproducibility checks, not independent stochastic evidence. This is why the tiny near-threshold changes are not promoted.

## Consequence

The result falsifies the claim that one low-KL LoRA update with either of these two scalar reward interventions is sufficient. It does not show that Task2 is impossible. The next locomotion experiment must change the optimization information—not continue either failed weight or sweep these coefficients—and jointly address posture tails, contact support, and the moving-to-stop basin. Any new method must fresh-start from Stage219 and keep the same source-retention gates.

Machine-readable result: `reports/retarget/x2_native_posture_phase60_result.json`.
