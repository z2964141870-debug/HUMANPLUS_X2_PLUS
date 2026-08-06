# Round 3 gait filter: FBP versus LBP projection at 25 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 4/4 | 0/4 | 54.31 | 120.63 | 49.96 | 523.18 | 189.07 | 0.627 |
| `LBP` | 4/4 | 4/4 | 0/4 | 51.74 | 117.66 | 46.55 | 367.51 | 167.97 | 0.645 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 58.72/127.88 | 725.13 | 220.41 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 48.72/106.55 | 269.02 | 153.42 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 53.90/112.92 | 364.96 | 155.28 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 55.88/125.12 | 182.23 | 156.68 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 50.31/114.45 | 313.19 | 173.26 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 48.78/103.74 | 293.57 | 149.23 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 53.36/119.00 | 423.49 | 166.79 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 54.52/120.39 | 142.59 | 163.36 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`True`, wrist mean `-4.72%`, wrist p95 `-2.46%`, upper mean `-6.83%`, anchor p95 `-29.75%`.
