# Stage264 new-machine replay

- Full functional gate: **22/24**.
- Raw external JSON traces: **80.8 MiB**.
- Complete matrix: **True**; no physics failure was retried or overwritten.

| speed | motion | pass | mean forward | mean yaw | signed pitch mean |
|---|---|---:|---:|---:|---:|
| medium | straight | 6/6 | 1.186 m | -0.002 rad | -0.197 rad |
| medium | turn_right | 3/3 | 1.273 m | 0.329 rad | -0.179 rad |
| medium | turn_left | 3/3 | 1.228 m | -0.296 rad | -0.188 rad |
| low | straight | 5/6 | 1.135 m | 0.007 rad | -0.183 rad |
| low | turn_right | 2/3 | 1.199 m | 0.346 rad | -0.160 rad |
| low | turn_left | 3/3 | 1.141 m | -0.270 rad | -0.180 rad |

## Failures

| case | passed phases | first violation | stop z min | stop tilt max | handoff action jump |
|---|---|---:|---:|---:|---:|
| stage264_low_straight_r4 | startup/move | 1.60 s | 0.117 m | 1.547 rad | 1.985 |
| stage264_low_turn_right_r3 | startup/move | 1.90 s | 0.117 m | 1.544 rad | 1.763 |

The historical 24/24 claim is not reproduced: the no-retry new-machine gate is 22/24. Both failures pass locomotion and fail after the direct moving-to-stationary policy handoff. Negative signed pitch also keeps Task 2 open.
