# Round 3 gait delay: FBP versus LBP projection at 5 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 4/4 | 0/4 | 67.38 | 153.95 | 55.73 | 486.31 | 187.12 | 0.639 |
| `LBP` | 4/4 | 4/4 | 0/4 | 69.09 | 160.96 | 54.89 | 383.51 | 163.03 | 0.638 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 64.89/172.56 | 314.56 | 188.61 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 70.01/142.87 | 500.11 | 176.22 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 68.03/152.55 | 441.38 | 161.91 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 66.58/142.93 | 655.16 | 222.89 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 64.39/166.03 | 389.54 | 150.63 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 68.49/142.85 | 203.17 | 190.57 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 71.38/156.43 | 413.95 | 163.96 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 72.13/164.37 | 192.57 | 157.19 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`True`, wrist mean `+2.55%`, wrist p95 `+4.55%`, upper mean `-1.50%`, anchor p95 `-21.14%`.
