# Original Gait Ideal-Domain Upper Metrics

Baseline: `B0`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | 4/4 | 4/4 | 0/4 | 42.75 | 73.23 | 41.44 | 597.32 | 130.72 | 0.645 |
| `E1-I25` | 4/4 | 4/4 | 0/4 | 43.18 | 72.34 | 42.04 | 341.97 | 136.60 | 0.643 |

## Per-motion

### B0

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 40.84/71.66 | 664.45 | 114.39 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 41.63/71.71 | 275.87 | 119.84 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 42.58/72.17 | 313.42 | 135.88 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 45.96/79.52 | 138.20 | 143.80 |

### E1-I25

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 241 | True | True | False | 42.61/76.96 | 419.60 | 144.09 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | 240 | True | True | False | 42.71/68.28 | 142.30 | 137.23 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 241 | True | True | False | 42.85/80.26 | 333.07 | 138.51 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | 240 | True | True | False | 44.55/68.72 | 256.02 | 134.10 |

## Baseline-relative preservation

- `E1-I25`: preserve@5%=`True`, wrist mean `+1.00%`, wrist p95 `-1.22%`, upper mean `+1.46%`, anchor p95 `-42.75%`.
