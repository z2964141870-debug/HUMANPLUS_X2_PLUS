# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 1/4 | 1/4 | 4/4 | 0.182 m | 1.834 |
| `filter` | 1/4 | 4/4 | 1/4 | 3/4 | 1/4 | 0.203 m | 0.725 |
| `delay` | 0/4 | 4/4 | 0/4 | 3/4 | 0/4 | 0.267 m | 3.712 |
| `noise` | 0/4 | 3/4 | 0/4 | 3/4 | 1/4 | 0.295 m | 1.808 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 6.44/0.99 | 0.380 | 0.60/0.76 | 7→6/6→8 | 0.148 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.52/0.97 | 0.104 | 0.69/0.55 | 5→4/2→4 | 0.189 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 1.97/0.94 | 0.156 | 0.51/0.50 | 6→5/7→4 | 0.177 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.40/0.95 | 0.086 | 0.52/0.52 | 2→5/5→5 | 0.161 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 0.37/0.07 | 0.366 | 0.48/0.55 | 7→5/6→5 | 0.243 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 0.37/0.67 | 0.148 | 0.75/0.67 | 5→4/2→4 | 0.209 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | -0.46/-0.20 | 0.207 | 0.59/0.64 | 6→5/7→6 | 0.231 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | pass | yes | 0.82/0.87 | 0.092 | 0.90/0.78 | 2→3/5→3 | 0.196 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -4.20/-0.88 | 0.237 | 0.47/0.55 | 7→9/6→6 | 0.216 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -5.31/-0.98 | 0.452 | 0.78/0.63 | 5→4/2→5 | 0.322 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.69/0.97 | 0.263 | 0.76/0.64 | 6→5/7→5 | 0.236 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.65/0.85 | 0.115 | 0.74/0.58 | 2→5/5→6 | 0.226 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | no | -0.42/-0.05 | 0.495 | 0.46/0.51 | 7→9/6→7 | 0.749 | 191 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 3.67/0.90 | 0.253 | 0.70/0.57 | 5→3/2→4 | 0.281 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.41/0.81 | 0.332 | 0.88/0.76 | 6→5/7→5 | 0.204 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.74/0.97 | 0.098 | 0.85/0.73 | 2→4/5→4 | 0.190 | — |
