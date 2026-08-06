# X2 诚实步态进度门禁

该门禁同时要求世界 root 前进、方向、接触时序、局部足端与不倒地；不再允许原地站立通过。

| condition | pass | stable | progress | contact | foot | root XY RMSE mean | mean absolute progress-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ideal` | 0/4 | 4/4 | 1/4 | 1/4 | 4/4 | 0.191 m | 1.335 |
| `filter` | 0/4 | 4/4 | 1/4 | 4/4 | 0/4 | 0.144 m | 1.686 |
| `delay` | 0/4 | 2/4 | 0/4 | 3/4 | 0/4 | 0.402 m | 5.034 |
| `noise` | 0/4 | 4/4 | 0/4 | 3/4 | 0/4 | 0.253 m | 2.750 |

## ideal

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 2.52/0.97 | 0.273 | 0.53/0.60 | 7→5/6→5 | 0.158 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 0.90/0.92 | 0.078 | 0.67/0.52 | 5→6/2→4 | 0.153 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.01/0.85 | 0.324 | 0.59/0.53 | 6→4/7→4 | 0.155 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 1.71/1.00 | 0.090 | 0.70/0.57 | 2→6/5→3 | 0.160 | — |

## filter

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | -1.54/-0.97 | 0.137 | 0.57/0.56 | 7→11/6→6 | 0.216 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.10/0.97 | 0.133 | 0.75/0.74 | 5→4/2→4 | 0.210 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.74/0.85 | 0.231 | 0.85/0.77 | 6→4/7→5 | 0.232 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.62/1.00 | 0.078 | 0.90/0.73 | 2→4/5→3 | 0.204 | — |

## delay

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | no | 12.76/0.91 | 0.829 | 0.56/0.69 | 7→10/6→10 | 0.694 | 186 |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | 2.00/0.72 | 0.202 | 0.74/0.68 | 5→3/2→4 | 0.260 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 3.22/0.92 | 0.237 | 0.75/0.62 | 6→7/7→8 | 0.228 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | no | -4.16/-0.61 | 0.341 | 0.58/0.53 | 2→7/5→9 | 0.571 | 233 |

## noise

| motion | result | stable | progress ratio/cos | root RMSE | contact BA L/R | cycles ref→meas L/R | foot max | first fall |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80` | fail | yes | 7.31/0.91 | 0.454 | 0.71/0.86 | 7→6/6→6 | 0.202 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77` | fail | yes | -0.09/-0.08 | 0.092 | 0.64/0.51 | 5→3/2→3 | 0.256 | — |
| `official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror` | fail | yes | 4.25/0.86 | 0.361 | 0.87/0.75 | 6→5/7→6 | 0.203 | — |
| `official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror` | fail | yes | 0.65/0.58 | 0.107 | 0.68/0.62 | 2→5/5→7 | 0.229 | — |
