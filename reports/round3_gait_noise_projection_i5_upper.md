# Round 3 gait noise: FBP versus LBP projection at 5 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 4/4 | 0/4 | 68.41 | 161.75 | 56.29 | 506.30 | 190.94 | 0.597 |
| `LBP` | 4/4 | 4/4 | 0/4 | 71.07 | 158.84 | 57.80 | 646.74 | 174.03 | 0.634 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 65.33/172.76 | 455.62 | 174.92 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 69.24/162.37 | 441.59 | 222.55 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 67.73/150.10 | 563.21 | 168.41 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 71.35/160.83 | 205.41 | 179.72 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 70.20/173.38 | 548.78 | 161.47 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 67.12/138.21 | 362.90 | 210.79 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 75.12/156.16 | 695.81 | 161.64 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 71.83/161.24 | 209.64 | 173.21 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`True`, wrist mean `+3.89%`, wrist p95 `-1.80%`, upper mean `+2.68%`, anchor p95 `+27.74%`.
