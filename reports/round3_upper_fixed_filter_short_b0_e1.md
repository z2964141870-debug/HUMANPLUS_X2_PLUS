# Fixed-Feet Filter-Domain 5.2s Causal Panel

Baseline: `B0`. Only the first frame-zero episode per motion is counted.

| checkpoint | complete | stable | strict | wrist mean mm | wrist p95 mm | upper mean mm | anchor p95 mm | foot p95 mm | root z min m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | 0/3 | 3/3 | 0/3 | 55.87 | 105.24 | 39.36 | 55.53 | 54.94 | 0.644 |
| `E1-I25` | 0/3 | 3/3 | 0/3 | 56.87 | 104.41 | 40.00 | 52.23 | 54.72 | 0.643 |

## Per-motion

### B0

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `balance_shift__balance_shift_20260708_211847` | 260 | False | True | False | 61.42/112.61 | 48.21 | 47.95 |
| `stand__stand_20260708_211440` | 260 | False | True | False | 44.94/79.59 | 42.50 | 46.46 |
| `wave__wave_20260708_211740` | 260 | False | True | False | 61.25/112.52 | 69.01 | 67.06 |

### E1-I25

| motion | samples | done | stable | strict | wrist mean/p95 mm | anchor p95 mm | foot p95 mm |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `balance_shift__balance_shift_20260708_211847` | 260 | False | True | False | 63.89/106.39 | 44.85 | 44.76 |
| `stand__stand_20260708_211440` | 260 | False | True | False | 48.77/76.89 | 43.22 | 44.67 |
| `wave__wave_20260708_211740` | 260 | False | True | False | 57.94/115.93 | 63.49 | 64.77 |

## Baseline-relative preservation

- `E1-I25`: preserve@5%=`True`, wrist mean `+1.78%`, wrist p95 `-0.79%`, upper mean `+1.64%`, anchor p95 `-5.94%`.
