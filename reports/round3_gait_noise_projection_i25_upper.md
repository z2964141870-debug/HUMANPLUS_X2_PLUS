# Round 3 gait noise: FBP versus LBP projection at 25 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 3/4 | 0/4 | 104.46 | 363.02 | 96.25 | 658.69 | 336.75 | 0.182 |
| `LBP` | 4/4 | 4/4 | 0/4 | 68.65 | 153.58 | 56.94 | 657.05 | 162.97 | 0.639 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | False | False | 198.25/759.78 | 1183.26 | 686.57 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 75.82/169.35 | 598.58 | 196.23 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 71.78/137.16 | 506.91 | 153.32 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 71.74/157.89 | 192.44 | 147.03 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 68.20/170.85 | 842.03 | 161.99 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 66.35/132.72 | 178.47 | 193.71 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 70.88/134.50 | 571.27 | 148.30 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 69.15/148.13 | 186.79 | 163.54 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`True`, wrist mean `-34.29%`, wrist p95 `-57.69%`, upper mean `-40.84%`, anchor p95 `-0.25%`.
