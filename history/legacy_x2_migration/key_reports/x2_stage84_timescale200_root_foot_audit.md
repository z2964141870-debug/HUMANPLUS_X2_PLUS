# X2 Reference Root–Foot Consistency Audit

- motions: 2
- MJCF: `/home/humanplus/x2_teleop_final/assets/agibot_x2/x2_ultra.xml`
- stance height margin: 0.025 m
- stance vertical-speed cap: 0.250 m/s

This is a ranking diagnostic, not a pass/fail gate. Stance is inferred from motion-local foot height and vertical speed.

| rank | motion | slip p95 (m/s) | stance excursion (m) | root accel p95 (m/s²) | single stance | COM outside L/R | LIPM ZMP outside L/R | score |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | 0.7290 | 0.1634 | 4.0775 | 0.696 | 0.992/0.819 | 0.832/0.867 | 9.148 |
| 2 | `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | 0.7290 | 0.1634 | 4.0775 | 0.696 | 0.819/1.000 | 0.867/0.840 | 9.148 |
