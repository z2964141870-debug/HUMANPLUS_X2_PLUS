# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 2/12 | 12/12 | 2/12 | 7/12 | 12/12 | 0.193 m | 2.042 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__wave__wave_20260708_211740` | fail | yes | 2.28/1.00 | 0.104 | 0.78/0.57 | 2→2/5→3 | 0.154 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__wave__wave_20260708_211740` | fail | yes | 6.59/0.88 | 0.460 | 0.56/0.60 | 7→6/6→5 | 0.162 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__stand__stand_20260708_211440` | fail | yes | 3.82/0.95 | 0.259 | 0.59/0.55 | 7→8/6→7 | 0.155 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__wave__wave_20260708_211740` | pass | yes | 1.39/0.99 | 0.074 | 0.59/0.63 | 5→6/2→3 | 0.171 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__stand__stand_20260708_211440` | fail | yes | 2.54/0.92 | 0.174 | 0.52/0.51 | 6→6/7→6 | 0.188 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 2.89/1.00 | 0.158 | 0.72/0.55 | 2→5/5→3 | 0.148 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__wave__wave_20260708_211740` | fail | yes | 3.08/0.89 | 0.199 | 0.56/0.52 | 6→6/7→7 | 0.160 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__stand__stand_20260708_211440` | fail | yes | 2.39/1.00 | 0.132 | 0.69/0.55 | 2→3/5→3 | 0.155 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 3.14/0.82 | 0.208 | 0.58/0.53 | 6→4/7→5 | 0.173 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 4.73/0.97 | 0.355 | 0.48/0.62 | 7→7/6→5 | 0.155 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 2.40/0.99 | 0.111 | 0.63/0.66 | 5→4/2→3 | 0.166 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__stand__stand_20260708_211440` | pass | yes | 1.25/0.98 | 0.076 | 0.56/0.64 | 5→5/2→3 | 0.167 | — |
