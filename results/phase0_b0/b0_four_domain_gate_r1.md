# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 1/4 | 4/4 | 1/4 | 3/4 | 4/4 | 0.193 m | 1.850 |
| `filter` | 0/4 | 4/4 | 1/4 | 2/4 | 0/4 | 0.140 m | 1.283 |
| `delay` | 0/4 | 3/4 | 1/4 | 4/4 | 0/4 | 0.364 m | 3.464 |
| `noise` | 0/4 | 4/4 | 1/4 | 2/4 | 0/4 | 0.252 m | 2.809 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 5.37/0.99 | 0.313 | 0.60/0.77 | 7→8/6→4 | 0.152 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | pass | yes | 1.36/0.98 | 0.104 | 0.60/0.60 | 5→5/2→7 | 0.153 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.97/0.86 | 0.260 | 0.52/0.51 | 6→4/7→5 | 0.159 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.69/0.99 | 0.097 | 0.73/0.59 | 2→3/5→3 | 0.156 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -1.52/-0.67 | 0.135 | 0.47/0.58 | 7→12/6→5 | 0.200 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 0.22/0.40 | 0.102 | 0.72/0.52 | 5→3/2→3 | 0.200 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.47/0.94 | 0.232 | 0.75/0.65 | 6→7/7→6 | 0.223 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.36/1.00 | 0.091 | 0.71/0.59 | 2→3/5→5 | 0.206 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 1.36/0.99 | 0.095 | 0.60/0.76 | 7→8/6→5 | 0.254 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | no | -7.58/-0.95 | 0.913 | 0.59/0.78 | 5→4/2→6 | 0.746 | 156 |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.07/0.99 | 0.252 | 0.77/0.63 | 6→6/7→6 | 0.230 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | -1.84/-0.76 | 0.195 | 0.64/0.64 | 2→1/5→2 | 0.235 | — |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -6.76/-0.95 | 0.399 | 0.40/0.48 | 7→12/6→5 | 0.233 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.72/0.69 | 0.239 | 0.66/0.53 | 5→2/2→4 | 0.262 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 0.68/0.71 | 0.218 | 0.77/0.71 | 6→6/7→5 | 0.205 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 2.43/0.99 | 0.151 | 0.92/0.77 | 2→5/5→3 | 0.212 | — |
