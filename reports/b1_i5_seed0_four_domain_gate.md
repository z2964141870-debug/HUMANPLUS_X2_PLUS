# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 0/4 | 1/4 | 3/4 | 0.198 m | 1.997 |
| `filter` | 0/4 | 4/4 | 0/4 | 4/4 | 2/4 | 0.188 m | 1.536 |
| `delay` | 0/4 | 4/4 | 1/4 | 4/4 | 2/4 | 0.196 m | 1.938 |
| `noise` | 0/4 | 4/4 | 1/4 | 3/4 | 0/4 | 0.240 m | 3.151 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 3.55/0.94 | 0.222 | 0.62/0.46 | 7→9/6→5 | 0.213 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.31/1.00 | 0.145 | 0.65/0.55 | 5→5/2→4 | 0.188 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.57/0.81 | 0.352 | 0.70/0.54 | 6→5/7→4 | 0.153 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.56/0.99 | 0.071 | 0.61/0.57 | 2→4/5→2 | 0.180 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 2.17/1.00 | 0.202 | 0.62/0.81 | 7→6/6→4 | 0.209 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -0.06/-0.31 | 0.193 | 0.77/0.77 | 5→5/2→4 | 0.168 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.74/0.75 | 0.256 | 0.89/0.73 | 6→5/7→5 | 0.187 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -0.17/-0.49 | 0.100 | 0.85/0.71 | 2→3/5→4 | 0.222 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 4.31/0.82 | 0.314 | 0.61/0.78 | 7→6/6→5 | 0.199 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 0.60/1.00 | 0.079 | 0.71/0.63 | 5→4/2→5 | 0.219 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.84/0.80 | 0.290 | 0.86/0.73 | 6→5/7→6 | 0.213 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.20/0.99 | 0.101 | 0.92/0.76 | 2→4/5→3 | 0.183 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -9.94/-1.00 | 0.467 | 0.46/0.62 | 7→8/6→9 | 0.329 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.64/0.96 | 0.179 | 0.72/0.69 | 5→2/2→4 | 0.230 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 0.16/0.24 | 0.211 | 0.72/0.66 | 6→4/7→4 | 0.204 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.81/0.91 | 0.104 | 0.90/0.69 | 2→4/5→4 | 0.224 | — |
