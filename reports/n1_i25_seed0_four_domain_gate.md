# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 0/4 | 2/4 | 4/4 | 0.190 m | 1.668 |
| `filter` | 0/4 | 4/4 | 0/4 | 4/4 | 1/4 | 0.238 m | 2.685 |
| `delay` | 0/4 | 4/4 | 1/4 | 4/4 | 1/4 | 0.158 m | 1.398 |
| `noise` | 0/4 | 4/4 | 0/4 | 2/4 | 1/4 | 0.195 m | 1.965 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 2.86/0.82 | 0.262 | 0.59/0.53 | 7→11/6→8 | 0.185 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.06/0.94 | 0.118 | 0.59/0.61 | 5→6/2→4 | 0.171 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.92/0.84 | 0.269 | 0.63/0.51 | 6→6/7→4 | 0.177 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.82/0.99 | 0.112 | 0.72/0.57 | 2→5/5→3 | 0.159 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 5.90/0.93 | 0.440 | 0.62/0.74 | 7→7/6→5 | 0.178 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.82/0.90 | 0.179 | 0.69/0.81 | 5→4/2→4 | 0.219 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.37/1.00 | 0.203 | 0.80/0.71 | 6→4/7→5 | 0.220 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.65/0.99 | 0.130 | 0.78/0.68 | 2→5/5→4 | 0.218 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 1.81/0.84 | 0.174 | 0.61/0.81 | 7→7/6→5 | 0.203 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 3.58/1.00 | 0.185 | 0.74/0.92 | 5→6/2→5 | 0.183 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.86/0.96 | 0.187 | 0.74/0.64 | 6→6/7→6 | 0.228 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.34/0.78 | 0.088 | 0.92/0.64 | 2→3/5→5 | 0.213 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 2.98/1.00 | 0.175 | 0.50/0.74 | 7→8/6→4 | 0.230 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -0.85/-0.85 | 0.170 | 0.61/0.52 | 5→4/2→5 | 0.220 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 0.22/0.10 | 0.202 | 0.61/0.56 | 6→6/7→5 | 0.225 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 4.24/1.00 | 0.234 | 0.90/0.69 | 2→4/5→5 | 0.177 | — |
