# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 1/4 | 0/4 | 4/4 | 0.257 m | 3.155 |
| `filter` | 1/4 | 4/4 | 1/4 | 4/4 | 3/4 | 0.204 m | 1.911 |
| `delay` | 0/4 | 4/4 | 0/4 | 4/4 | 1/4 | 0.184 m | 1.827 |
| `noise` | 0/4 | 3/4 | 0/4 | 3/4 | 2/4 | 0.228 m | 2.681 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 8.15/0.92 | 0.538 | 0.54/0.62 | 7→6/6→6 | 0.149 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -0.12/-0.38 | 0.067 | 0.62/0.43 | 5→8/2→5 | 0.171 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.86/0.82 | 0.369 | 0.65/0.54 | 6→5/7→4 | 0.151 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.51/0.78 | 0.056 | 0.47/0.53 | 2→4/5→3 | 0.176 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 5.85/0.93 | 0.393 | 0.62/0.72 | 7→7/6→5 | 0.187 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | pass | yes | 1.17/0.97 | 0.107 | 0.78/0.81 | 5→4/2→4 | 0.180 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.19/0.87 | 0.156 | 0.76/0.61 | 6→5/7→5 | 0.191 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.57/0.33 | 0.161 | 0.81/0.68 | 2→4/5→5 | 0.245 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 4.32/0.96 | 0.338 | 0.60/0.74 | 7→7/6→5 | 0.207 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.41/1.00 | 0.135 | 0.74/0.68 | 5→4/2→6 | 0.227 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 1.89/1.00 | 0.094 | 0.69/0.66 | 6→5/7→8 | 0.211 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.69/0.98 | 0.170 | 0.90/0.63 | 2→3/5→5 | 0.196 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 0.46/0.46 | 0.106 | 0.56/0.67 | 7→8/6→4 | 0.201 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | no | 5.50/1.00 | 0.256 | 0.66/0.50 | 5→4/2→3 | 0.474 | 238 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 5.01/0.84 | 0.411 | 0.86/0.75 | 6→5/7→6 | 0.200 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.68/0.99 | 0.140 | 0.85/0.70 | 2→4/5→5 | 0.178 | — |
