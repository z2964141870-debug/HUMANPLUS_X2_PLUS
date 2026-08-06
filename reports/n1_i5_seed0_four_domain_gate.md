# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 1/4 | 4/4 | 2/4 | 1/4 | 3/4 | 0.126 m | 0.926 |
| `filter` | 1/4 | 4/4 | 1/4 | 4/4 | 2/4 | 0.194 m | 2.195 |
| `delay` | 0/4 | 3/4 | 0/4 | 3/4 | 2/4 | 0.417 m | 4.796 |
| `noise` | 0/4 | 3/4 | 0/4 | 3/4 | 1/4 | 0.345 m | 4.252 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 0.23/0.98 | 0.087 | 0.54/0.47 | 7→10/6→6 | 0.210 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | pass | yes | 1.45/0.97 | 0.114 | 0.59/0.57 | 5→7/2→5 | 0.159 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.47/0.83 | 0.230 | 0.60/0.51 | 6→4/7→4 | 0.171 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.99/0.95 | 0.071 | 0.53/0.51 | 2→5/5→4 | 0.167 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 6.87/0.93 | 0.427 | 0.64/0.85 | 7→6/6→5 | 0.167 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | pass | yes | 1.10/0.95 | 0.113 | 0.77/0.79 | 5→4/2→4 | 0.172 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.58/0.97 | 0.137 | 0.86/0.67 | 6→4/7→7 | 0.226 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -0.23/-0.67 | 0.098 | 0.75/0.68 | 2→3/5→3 | 0.231 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | no | 12.92/0.90 | 0.824 | 0.54/0.78 | 7→11/6→7 | 0.648 | 186 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -1.89/-0.89 | 0.347 | 0.67/0.56 | 5→7/2→5 | 0.197 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 1.66/0.73 | 0.219 | 0.64/0.57 | 6→7/7→5 | 0.220 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 4.72/0.98 | 0.279 | 0.90/0.63 | 2→3/5→4 | 0.197 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 8.21/0.91 | 0.481 | 0.70/0.83 | 7→6/6→7 | 0.200 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | no | 5.42/0.99 | 0.266 | 0.65/0.45 | 5→4/2→6 | 0.606 | 216 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 5.68/0.70 | 0.464 | 0.87/0.74 | 6→5/7→5 | 0.205 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.70/0.78 | 0.168 | 0.67/0.58 | 2→4/5→5 | 0.222 | — |
