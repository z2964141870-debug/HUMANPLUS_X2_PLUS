# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | 0/4 | 4/4 | 0/4 | 2/4 | 4/4 | 0.188 m | 2.303 |
| `materialized` | 0/4 | 4/4 | 0/4 | 2/4 | 4/4 | 0.178 m | 1.655 |

## B0

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 6.36/0.96 | 0.394 | 0.57/0.74 | 7→8/6→4 | 0.147 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.10/1.00 | 0.116 | 0.63/0.62 | 5→9/2→3 | 0.186 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 2.94/0.87 | 0.178 | 0.57/0.50 | 6→6/7→4 | 0.186 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.19/0.38 | 0.064 | 0.43/0.51 | 2→5/5→3 | 0.168 | — |

## materialized

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 2.19/0.91 | 0.117 | 0.60/0.52 | 7→8/6→7 | 0.185 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 1.89/0.96 | 0.142 | 0.63/0.60 | 5→7/2→4 | 0.194 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.78/0.83 | 0.359 | 0.66/0.54 | 6→5/7→4 | 0.144 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.76/1.00 | 0.095 | 0.63/0.58 | 2→4/5→3 | 0.162 | — |
