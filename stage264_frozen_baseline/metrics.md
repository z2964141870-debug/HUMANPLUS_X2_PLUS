# Stage264 frozen baseline metrics

Historical evidence is SHA-bound and restored on the new machine. No new physics or optimizer steps were run.

| motion | signed pitch mean | p05–p95 | yaw progress |
|---|---:|---:|---:|
| straight | -11.29° | -13.15° … -8.06° | 4.10° |
| right | -10.20° | -12.25° … -7.98° | 19.37° |
| left | -10.72° | -12.36° … -7.92° | -17.85° |

- Historical nominal start/walk/turn/stop: **24/24**.
- PD×fixed/fast-upper straight robustness: **14/18**.
- Representative straight speed: **0.295 m/s**.
- Realized stance-slip p95: left **0.425 m/s**, right **0.392 m/s**; strict 0.20 m/s gate fails.
- The 50 fps side/turn review videos are part of the hash-bound evidence inventory.
- New-machine 24/24 replay remains false until the exact gait-template asset is restored.
