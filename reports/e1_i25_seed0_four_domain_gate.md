# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 1/4 | 4/4 | 1/4 | 3/4 | 4/4 | 0.150 m | 1.499 |
| `filter` | 0/4 | 2/4 | 0/4 | 3/4 | 2/4 | 0.311 m | 3.179 |
| `delay` | 0/4 | 3/4 | 1/4 | 3/4 | 1/4 | 0.182 m | 2.382 |
| `noise` | 0/4 | 3/4 | 1/4 | 2/4 | 0/4 | 0.176 m | 1.745 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -2.12/-0.90 | 0.226 | 0.62/0.60 | 7→10/6→8 | 0.180 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | pass | yes | 1.09/0.99 | 0.068 | 0.64/0.59 | 5→5/2→5 | 0.158 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.68/0.84 | 0.184 | 0.54/0.47 | 6→5/7→5 | 0.179 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.11/1.00 | 0.123 | 0.71/0.58 | 2→4/5→3 | 0.155 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | no | -6.11/-0.87 | 0.405 | 0.64/0.71 | 7→10/6→8 | 0.575 | 212 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.51/0.99 | 0.078 | 0.78/0.81 | 5→6/2→4 | 0.182 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | no | -2.87/-0.30 | 0.649 | 0.52/0.56 | 6→8/7→7 | 0.630 | 173 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.23/0.97 | 0.112 | 0.82/0.75 | 2→4/5→5 | 0.183 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | no | -3.65/-0.91 | 0.284 | 0.51/0.59 | 7→9/6→7 | 0.460 | 239 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.14/0.89 | 0.102 | 0.70/0.64 | 5→4/2→5 | 0.222 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.69/0.96 | 0.239 | 0.74/0.63 | 6→5/7→6 | 0.223 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.05/0.96 | 0.105 | 0.85/0.72 | 2→4/5→4 | 0.182 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -0.51/-0.93 | 0.132 | 0.53/0.61 | 7→9/6→6 | 0.201 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | no | 5.27/0.90 | 0.345 | 0.69/0.47 | 5→6/2→4 | 0.670 | 215 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 0.14/0.18 | 0.131 | 0.70/0.63 | 6→6/7→6 | 0.210 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.34/0.97 | 0.097 | 0.88/0.71 | 2→4/5→5 | 0.205 | — |
