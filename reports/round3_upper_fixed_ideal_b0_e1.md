# Fixed-Feet Upper-Body Capability Panel

Baseline: `B0`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | 3/3 | 3/3 | 2/3 | 51.90 | 121.67 | 37.52 | 32.13 | 35.94 | 0.643 |
| `E1-I25` | 3/3 | 3/3 | 2/3 | 53.85 | 119.69 | 37.45 | 28.81 | 33.10 | 0.644 |

## Per-motion

### B0

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `balance_shift__balance_shift_20260708_211847` | 1252 | True | True | True | 48.57/73.73 | 27.24 | 35.93 |
| `stand__stand_20260708_211440` | 1949 | True | True | True | 42.61/55.99 | 24.70 | 31.23 |
| `wave__wave_20260708_211740` | 1949 | True | True | False | 63.32/156.63 | 35.38 | 38.43 |

### E1-I25

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `balance_shift__balance_shift_20260708_211847` | 1252 | True | True | True | 52.00/73.19 | 24.11 | 32.30 |
| `stand__stand_20260708_211440` | 1949 | True | True | True | 48.01/58.78 | 22.08 | 27.96 |
| `wave__wave_20260708_211740` | 1949 | True | True | False | 60.87/151.32 | 31.05 | 34.80 |

## Baseline-relative preservation

- `E1-I25`: preserve@5%=`True`, wrist mean `+3.75%`, wrist p95 `-1.63%`, upper mean `-0.18%`, anchor p95 `-10.36%`.
