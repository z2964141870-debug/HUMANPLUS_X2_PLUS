# X2 官方 MuJoCo 起步—行走—转向—停车门禁

- 域：`aimdk_x2_v1_official_mujoco`
- 矩阵完整：`True`
- 矩阵通过：`True`

| 技能 | 通过/运行 | 通过率 |
|---|---:|---:|
| straight | 3/3 | 100% |
| turn_left | 3/3 | 100% |
| turn_right | 3/3 | 100% |

每次运行同时要求：站立、起步前 1 秒、移动语义和停车全部通过。起步门禁防止用末段结果掩盖开局下沉、反退或大倾斜。

| 运行 | 起步 | 移动 | 停车 | 全门 | 前进(m) | 横漂(m) | 航向进展(rad) | 停稳(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage219_closed_loop_gate_v5_straight_r1 | True | True | True | True | 0.735 | -0.209 | 0.012 | 0.56 |
| stage219_closed_loop_gate_v5_straight_r2 | True | True | True | True | 0.826 | -0.243 | 0.041 | 0.60 |
| stage219_closed_loop_gate_v5_straight_r3 | True | True | True | True | 0.754 | -0.147 | 0.105 | 0.74 |
| stage219_closed_loop_gate_v5_turn_left_r1 | True | True | True | True | 0.969 | 0.032 | -0.397 | 1.00 |
| stage219_closed_loop_gate_v5_turn_left_r2 | True | True | True | True | 0.986 | 0.017 | -0.416 | 0.82 |
| stage219_closed_loop_gate_v5_turn_left_r3 | True | True | True | True | 0.997 | 0.044 | -0.356 | 0.64 |
| stage219_closed_loop_gate_v5_turn_right_r1 | True | True | True | True | 1.000 | -0.042 | 0.335 | 0.62 |
| stage219_closed_loop_gate_v5_turn_right_r2 | True | True | True | True | 0.955 | -0.001 | 0.446 | 1.02 |
| stage219_closed_loop_gate_v5_turn_right_r3 | True | True | True | True | 0.981 | 0.005 | 0.471 | 0.60 |
