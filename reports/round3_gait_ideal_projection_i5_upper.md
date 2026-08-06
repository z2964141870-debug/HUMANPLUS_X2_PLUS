# Round 3 gait ideal: FBP versus LBP projection at 5 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 4/4 | 0/4 | 41.75 | 68.22 | 41.42 | 258.73 | 142.56 | 0.647 |
| `LBP` | 4/4 | 4/4 | 0/4 | 42.49 | 70.31 | 42.17 | 272.57 | 141.78 | 0.644 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 39.70/67.09 | 238.25 | 146.91 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 41.04/67.24 | 172.23 | 140.10 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 42.39/71.64 | 297.87 | 137.11 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 43.87/67.51 | 119.05 | 142.06 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 41.09/73.00 | 262.59 | 142.76 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 42.20/69.32 | 118.80 | 142.59 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 42.60/72.49 | 363.63 | 136.70 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 44.07/66.95 | 126.27 | 139.20 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`True`, wrist mean `+1.77%`, wrist p95 `+3.07%`, upper mean `+1.81%`, anchor p95 `+5.35%`.
