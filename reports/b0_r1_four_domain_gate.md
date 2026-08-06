# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 0/4 | 2/4 | 4/4 | 0.188 m | 2.303 |
| `filter` | 1/4 | 4/4 | 1/4 | 4/4 | 3/4 | 0.147 m | 1.576 |
| `delay` | 0/4 | 4/4 | 0/4 | 3/4 | 0/4 | 0.348 m | 4.809 |
| `noise` | 0/4 | 4/4 | 1/4 | 3/4 | 2/4 | 0.265 m | 2.617 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 6.36/0.96 | 0.394 | 0.57/0.74 | 7→8/6→4 | 0.147 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.10/1.00 | 0.116 | 0.63/0.62 | 5→9/2→3 | 0.186 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.94/0.87 | 0.178 | 0.57/0.50 | 6→6/7→4 | 0.186 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.19/0.38 | 0.064 | 0.43/0.51 | 2→5/5→3 | 0.168 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 5.08/0.98 | 0.285 | 0.65/0.86 | 7→5/6→5 | 0.186 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.51/1.00 | 0.098 | 0.78/0.83 | 5→4/2→4 | 0.179 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.52/0.99 | 0.114 | 0.87/0.71 | 6→4/7→5 | 0.204 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | pass | yes | 0.80/0.96 | 0.090 | 0.88/0.76 | 2→3/5→4 | 0.194 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 10.66/0.93 | 0.642 | 0.64/0.78 | 7→6/6→6 | 0.247 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 3.15/0.99 | 0.185 | 0.76/0.72 | 5→4/2→6 | 0.228 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.40/0.81 | 0.294 | 0.86/0.76 | 6→5/7→5 | 0.234 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -3.01/-0.60 | 0.273 | 0.70/0.53 | 2→8/5→11 | 0.365 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 1.36/0.92 | 0.125 | 0.58/0.66 | 7→9/6→5 | 0.227 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -2.48/-0.99 | 0.335 | 0.70/0.46 | 5→4/2→3 | 0.174 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 7.27/0.90 | 0.491 | 0.88/0.77 | 6→5/7→6 | 0.204 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.64/0.61 | 0.109 | 0.66/0.64 | 2→5/5→7 | 0.174 | — |
