# Round 3 gait filter: FBP versus LBP projection at 5 steps

Baseline: `FBP`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FBP` | 4/4 | 4/4 | 0/4 | 51.60 | 119.42 | 45.00 | 679.05 | 158.45 | 0.640 |
| `LBP` | 4/4 | 4/4 | 0/4 | 51.42 | 122.86 | 44.68 | 428.91 | 149.89 | 0.641 |

## Per-motion

### FBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 48.71/135.87 | 721.36 | 148.16 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 49.70/105.44 | 229.53 | 119.58 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 53.35/120.47 | 318.35 | 188.48 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 54.66/114.44 | 337.85 | 178.87 |

### LBP

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 49.70/133.14 | 460.28 | 142.61 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 48.92/104.15 | 431.68 | 149.10 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 51.44/125.65 | 315.66 | 141.74 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 55.62/121.75 | 181.47 | 160.65 |

## Baseline-relative preservation

- `LBP`: preserve@5%=`True`, wrist mean `-0.35%`, wrist p95 `+2.88%`, upper mean `-0.71%`, anchor p95 `-36.84%`.
