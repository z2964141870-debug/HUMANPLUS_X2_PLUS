# Orin benchmark results

Date: 2026-08-25

Target: X2 SoC1 Orin NX, isolated offline benchmark

Command:

```bash
python benchmark_gmr.py --frames 220 --warmup 20 --max-iter 2
```

Input: deterministic smooth 220-frame SMPLight sequence. The first 20 frames are
excluded from timing. Each implementation receives the same sequence and starts
from a fresh GMR configuration.

| Implementation | Median | p95 | Mean |
| --- | ---: | ---: | ---: |
| Deployed baseline | 9.982 ms | 10.636 ms | 10.018 ms |
| Cached-limit call fix | 4.922 ms | 5.450 ms | 4.981 ms |
| Cached-limit fix + C++ preprocessing | 3.198 ms | 3.619 ms | 3.263 ms |

Numerical comparison against the deployed baseline trajectory:

| Implementation | Median abs qpos error | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Cached-limit call fix | 0 rad | 0 rad | 0 rad |
| C++ preprocessing | 5e-9 rad | 3.3e-8 rad | 5.7e-8 rad |

The standalone C++ preprocessing kernel was also checked on 1000 randomized
frames on both macOS arm64 and the Orin. Maximum position/quaternion component
error was `8.882e-16`.

Profiling the deployed baseline over 200 measured frames showed:

- 600 IK solves spent 1.104 s in inequality assembly.
- 600 accidental `ConfigurationLimit` constructions spent 0.789 s directly.
- DAQP solve wrappers spent only 0.090 s cumulatively.
- Python target preprocessing spent 0.544 s cumulatively.

The synthetic sequence establishes algorithmic equivalence and isolated speed.
An actual garment-frame replay should still be recorded and evaluated before the
native adapter is considered for integration into the shadow launcher.
