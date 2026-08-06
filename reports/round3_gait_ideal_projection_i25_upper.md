# Round 3 gait ideal: FBP versus LBP projection at 25 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 4/4 | 0/4 | 42.56 | 70.19 | 41.12 | 577.64 | 132.19 | 0.645 |
| `LBP` | 4/4 | 4/4 | 0/4 | 42.50 | 70.80 | 41.69 | 504.52 | 132.11 | 0.643 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 41.20/72.48 | 655.03 | 125.21 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 41.78/67.35 | 213.55 | 129.80 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 42.00/70.87 | 243.37 | 137.11 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 45.27/67.54 | 185.68 | 132.63 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 41.46/69.13 | 442.89 | 136.42 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 41.92/73.74 | 144.47 | 134.34 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 41.79/71.95 | 568.41 | 125.77 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 44.86/67.93 | 191.66 | 131.93 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`True`, wrist mean `-0.14%`, wrist p95 `+0.87%`, upper mean `+1.38%`, anchor p95 `-12.66%`.
