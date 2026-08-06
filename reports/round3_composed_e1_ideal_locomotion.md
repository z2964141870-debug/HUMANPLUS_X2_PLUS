# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/12 | 12/12 | 1/12 | 2/12 | 12/12 | 0.218 m | 2.696 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__wave__wave_20260708_211740` | fail | yes | 2.71/1.00 | 0.138 | 0.80/0.59 | 2→3/5→3 | 0.145 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__wave__wave_20260708_211740` | fail | yes | 7.36/0.90 | 0.479 | 0.50/0.67 | 7→6/6→6 | 0.145 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__stand__stand_20260708_211440` | fail | yes | 7.41/0.98 | 0.468 | 0.55/0.50 | 7→7/6→7 | 0.170 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__wave__wave_20260708_211740` | fail | yes | 1.51/1.00 | 0.086 | 0.60/0.51 | 5→6/2→4 | 0.181 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__stand__stand_20260708_211440` | fail | yes | 2.95/0.93 | 0.178 | 0.57/0.51 | 6→5/7→6 | 0.189 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 3.13/1.00 | 0.166 | 0.70/0.55 | 2→4/5→3 | 0.159 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__wave__wave_20260708_211740` | fail | yes | 2.03/0.82 | 0.146 | 0.51/0.49 | 6→6/7→5 | 0.192 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror__upper__stand__stand_20260708_211440` | fail | yes | 2.81/1.00 | 0.156 | 0.73/0.56 | 2→3/5→3 | 0.151 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 4.04/0.90 | 0.214 | 0.60/0.52 | 6→6/7→5 | 0.169 | — |
| `compose__official_x2__19_walk_forward_short_pulses_D__15p00_19p80__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 6.09/0.98 | 0.374 | 0.49/0.58 | 7→10/6→6 | 0.160 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__balance_shift__balance_shift_20260708_211847` | fail | yes | 2.94/0.99 | 0.145 | 0.56/0.52 | 5→6/2→3 | 0.141 | — |
| `compose__official_x2__19_walk_forward_short_pulses_B__10p97_15p77__upper__stand__stand_20260708_211440` | fail | yes | 0.62/0.91 | 0.062 | 0.65/0.46 | 5→8/2→5 | 0.150 | — |
