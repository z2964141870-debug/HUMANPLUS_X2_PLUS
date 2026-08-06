# Gait × Upper-Body Conditional Capability Panel

Baseline: `B0`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | 12/12 | 12/12 | 0/12 | 49.50 | 90.86 | 43.22 | 502.54 | 127.95 | 0.640 |
| `E1-I25` | 12/12 | 12/12 | 0/12 | 50.18 | 91.00 | 44.24 | 603.34 | 129.34 | 0.641 |

## Per-motion

### B0

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__wave__wave_20260708_211740` | 240 | True | True | False | 54.92/125.39 | 255.31 | 121.22 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__wave__wave_20260708_211740` | 241 | True | True | False | 52.27/110.56 | 824.89 | 131.55 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__stand__stand_20260708_211440` | 241 | True | True | False | 43.27/77.80 | 432.64 | 130.49 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__wave__wave_20260708_211740` | 240 | True | True | False | 56.53/96.74 | 165.08 | 117.30 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__stand__stand_20260708_211440` | 241 | True | True | False | 46.03/71.17 | 279.83 | 133.87 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__balance_shift__balance_shift_20260708_211847` | 240 | True | True | False | 49.40/73.77 | 357.53 | 127.62 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__wave__wave_20260708_211740` | 241 | True | True | False | 54.60/113.17 | 335.13 | 131.05 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__stand__stand_20260708_211440` | 240 | True | True | False | 46.71/67.20 | 291.16 | 129.96 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__balance_shift__balance_shift_20260708_211847` | 241 | True | True | False | 48.51/72.86 | 390.93 | 134.37 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__balance_shift__balance_shift_20260708_211847` | 241 | True | True | False | 44.57/80.92 | 578.25 | 121.73 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__balance_shift__balance_shift_20260708_211847` | 240 | True | True | False | 49.72/88.89 | 253.94 | 119.86 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__stand__stand_20260708_211440` | 240 | True | True | False | 47.46/82.33 | 166.22 | 119.10 |

### E1-I25

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__wave__wave_20260708_211740` | 240 | True | True | False | 53.03/118.70 | 323.54 | 123.20 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__wave__wave_20260708_211740` | 241 | True | True | False | 55.13/117.12 | 879.55 | 116.98 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__stand__stand_20260708_211440` | 241 | True | True | False | 44.64/73.98 | 794.17 | 131.54 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__wave__wave_20260708_211740` | 240 | True | True | False | 56.43/96.02 | 196.02 | 122.71 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__stand__stand_20260708_211440` | 241 | True | True | False | 47.18/73.04 | 296.24 | 138.92 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__balance_shift__balance_shift_20260708_211847` | 240 | True | True | False | 50.84/72.57 | 399.67 | 128.21 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__wave__wave_20260708_211740` | 241 | True | True | False | 51.50/109.23 | 240.40 | 140.00 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__stand__stand_20260708_211440` | 240 | True | True | False | 48.01/70.76 | 351.84 | 130.75 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__balance_shift__balance_shift_20260708_211847` | 241 | True | True | False | 49.53/75.61 | 427.10 | 137.75 |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__balance_shift__balance_shift_20260708_211847` | 241 | True | True | False | 46.57/90.12 | 617.11 | 127.85 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__balance_shift__balance_shift_20260708_211847` | 240 | True | True | False | 51.76/87.42 | 353.12 | 118.04 |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__stand__stand_20260708_211440` | 240 | True | True | False | 47.56/75.61 | 119.74 | 126.15 |

## Baseline-relative preservation

- `E1-I25`: preserve@5%=`True`, wrist mean `+1.38%`, wrist p95 `+0.16%`, upper mean `+2.35%`, anchor p95 `+20.06%`.
