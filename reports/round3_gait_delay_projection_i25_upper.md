# Round 3 gait delay: FBP versus LBP projection at 25 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 4/4 | 0/4 | 68.69 | 150.89 | 55.27 | 677.61 | 175.36 | 0.631 |
| `LBP` | 4/4 | 2/4 | 0/4 | 93.57 | 224.71 | 90.46 | 1246.30 | 411.28 | 0.167 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 66.65/142.23 | 570.23 | 177.07 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 73.45/196.02 | 850.96 | 189.90 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 65.50/144.59 | 395.91 | 170.30 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 69.17/154.88 | 229.80 | 159.70 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | False | False | 160.48/799.21 | 1764.78 | 632.11 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 67.24/142.93 | 398.00 | 159.84 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 68.40/152.32 | 382.34 | 167.12 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | False | False | 78.00/166.89 | 826.36 | 339.14 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`False`, wrist mean `+36.23%`, wrist p95 `+48.92%`, upper mean `+63.69%`, anchor p95 `+83.93%`.
