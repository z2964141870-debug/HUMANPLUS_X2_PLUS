# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 0/4 | 3/4 | 4/4 | 0.200 m | 1.801 |
| `filter` | 0/4 | 4/4 | 0/4 | 3/4 | 2/4 | 0.255 m | 3.182 |
| `delay` | 0/4 | 4/4 | 1/4 | 4/4 | 1/4 | 0.129 m | 0.758 |
| `noise` | 0/4 | 3/4 | 0/4 | 2/4 | 1/4 | 0.293 m | 2.859 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 4.29/0.89 | 0.343 | 0.57/0.56 | 7→8/6→5 | 0.156 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.01/1.00 | 0.110 | 0.59/0.67 | 5→7/2→3 | 0.175 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.78/0.81 | 0.231 | 0.59/0.50 | 6→8/7→5 | 0.143 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.12/1.00 | 0.114 | 0.71/0.56 | 2→5/5→3 | 0.147 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 8.19/0.82 | 0.483 | 0.50/0.61 | 7→9/6→7 | 0.301 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 3.42/1.00 | 0.191 | 0.79/0.90 | 5→3/2→4 | 0.179 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.14/0.77 | 0.191 | 0.79/0.64 | 6→5/7→5 | 0.187 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.02/0.01 | 0.154 | 0.80/0.70 | 2→4/5→4 | 0.247 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 0.44/1.00 | 0.119 | 0.56/0.76 | 7→8/6→6 | 0.228 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.51/0.79 | 0.191 | 0.72/0.64 | 5→4/2→4 | 0.258 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 1.05/0.85 | 0.123 | 0.80/0.66 | 6→7/7→7 | 0.210 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.91/1.00 | 0.082 | 0.92/0.80 | 2→5/5→3 | 0.179 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 3.32/0.63 | 0.257 | 0.52/0.63 | 7→9/6→4 | 0.324 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | no | 5.81/1.00 | 0.272 | 0.64/0.49 | 5→4/2→4 | 0.505 | 236 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.13/0.81 | 0.418 | 0.74/0.64 | 6→5/7→6 | 0.182 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.18/0.76 | 0.226 | 0.82/0.63 | 2→5/5→7 | 0.236 | — |
