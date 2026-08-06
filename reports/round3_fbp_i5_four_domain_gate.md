# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 1/4 | 4/4 | 2/4 | 2/4 | 4/4 | 0.112 m | 0.811 |
| `filter` | 1/4 | 4/4 | 1/4 | 4/4 | 2/4 | 0.226 m | 2.293 |
| `delay` | 0/4 | 4/4 | 0/4 | 2/4 | 0/4 | 0.240 m | 2.870 |
| `noise` | 0/4 | 4/4 | 0/4 | 3/4 | 0/4 | 0.217 m | 2.780 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 1.90/0.96 | 0.133 | 0.58/0.55 | 7→8/6→6 | 0.184 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | pass | yes | 1.34/1.00 | 0.078 | 0.61/0.70 | 5→5/2→3 | 0.193 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.75/0.86 | 0.174 | 0.57/0.51 | 6→6/7→4 | 0.190 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.75/0.96 | 0.061 | 0.53/0.54 | 2→4/5→4 | 0.171 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 6.45/0.92 | 0.456 | 0.62/0.80 | 7→5/6→5 | 0.186 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | pass | yes | 0.52/0.95 | 0.128 | 0.74/0.69 | 5→4/2→4 | 0.189 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | -1.24/-0.92 | 0.169 | 0.60/0.57 | 6→7/7→8 | 0.276 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -0.00/-0.00 | 0.151 | 0.84/0.71 | 2→4/5→3 | 0.242 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -0.87/-0.65 | 0.190 | 0.55/0.76 | 7→11/6→6 | 0.241 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 3.75/1.00 | 0.221 | 0.74/0.79 | 5→4/2→4 | 0.245 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.16/0.97 | 0.268 | 0.80/0.66 | 6→6/7→6 | 0.230 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -2.70/-0.59 | 0.281 | 0.56/0.49 | 2→7/5→9 | 0.304 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -2.87/-0.97 | 0.197 | 0.55/0.70 | 7→9/6→8 | 0.222 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.57/0.72 | 0.228 | 0.67/0.47 | 5→5/2→5 | 0.269 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 5.06/0.93 | 0.338 | 0.83/0.67 | 6→5/7→6 | 0.231 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -0.61/-0.66 | 0.106 | 0.60/0.59 | 2→5/5→5 | 0.244 | — |
