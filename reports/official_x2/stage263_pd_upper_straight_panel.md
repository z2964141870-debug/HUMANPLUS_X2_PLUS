# Stage263：官方 X2 MuJoCo 执行器参数 × 上肢扰动直行门禁

冻结 actor、步态模板和门槛；PD Kp/Kd 同比扰动，并交叉固定/快摆臂。

| condition | full gate | fallback | forward median | lateral abs max | tilt max | stop z min | stop settle median |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft0p9_fixed | 2/3 | 0 | 1.265 | 0.148 | 0.236 | 0.634 | 1.72 |
| soft0p9_fast | 3/3 | 0 | 1.240 | 0.125 | 0.251 | 0.631 | 1.94 |
| nominal1p0_fixed | 3/3 | 0 | 1.310 | 0.026 | 0.246 | 0.638 | 1.58 |
| nominal1p0_fast | 3/3 | 0 | 1.308 | 0.079 | 0.255 | 0.637 | 1.62 |
| stiff1p2_fixed | 0/3 | 0 | 1.288 | 0.143 | 0.254 | 0.632 | 0.48 |
| stiff1p2_fast | 3/3 | 0 | 1.277 | 0.070 | 0.257 | 0.642 | 0.54 |

结论：未通过。

该矩阵验证 PD 参数响应幅值，不等价于显式通信延迟或传感噪声；后二者仍需独立门禁。
