# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 2/4 | 0/4 | 4/4 | 0.121 m | 0.641 |
| `filter` | 1/4 | 4/4 | 1/4 | 4/4 | 3/4 | 0.190 m | 2.207 |
| `delay` | 0/4 | 4/4 | 1/4 | 4/4 | 2/4 | 0.176 m | 1.206 |
| `noise` | 0/4 | 4/4 | 0/4 | 2/4 | 0/4 | 0.252 m | 2.388 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 0.74/0.41 | 0.167 | 0.52/0.50 | 7→10/6→7 | 0.177 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.02/0.98 | 0.060 | 0.59/0.53 | 5→5/2→5 | 0.172 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.22/0.85 | 0.196 | 0.58/0.51 | 6→6/7→4 | 0.182 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.06/0.98 | 0.063 | 0.53/0.54 | 2→4/5→3 | 0.171 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 4.95/0.98 | 0.262 | 0.66/0.85 | 7→5/6→5 | 0.182 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -1.75/-1.00 | 0.243 | 0.80/0.76 | 5→4/2→4 | 0.211 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.72/0.76 | 0.163 | 0.68/0.64 | 6→4/7→5 | 0.162 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | pass | yes | 0.59/0.84 | 0.091 | 0.90/0.78 | 2→3/5→3 | 0.197 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 2.87/0.97 | 0.252 | 0.61/0.76 | 7→4/6→6 | 0.189 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.22/0.83 | 0.106 | 0.71/0.64 | 5→5/2→7 | 0.268 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.96/0.77 | 0.237 | 0.85/0.75 | 6→5/7→5 | 0.225 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.22/0.18 | 0.106 | 0.57/0.56 | 2→5/5→8 | 0.196 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -3.14/-0.95 | 0.255 | 0.52/0.56 | 7→7/6→6 | 0.234 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.78/0.65 | 0.207 | 0.65/0.46 | 5→3/2→4 | 0.256 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.39/0.71 | 0.433 | 0.84/0.73 | 6→5/7→5 | 0.252 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -0.24/-0.22 | 0.113 | 0.60/0.59 | 2→5/5→5 | 0.243 | — |
