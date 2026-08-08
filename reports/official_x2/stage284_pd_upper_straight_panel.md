# Stage284：官方 X2 MuJoCo 执行器参数 × 上肢扰动直行门禁

冻结 actor、步态模板和门槛；PD Kp/Kd 同比扰动，并交叉固定/快摆臂。

| condition | full gate | fallback | forward median | lateral abs max | tilt max | stop z min | stop settle median |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft0p9_fixed | 3/3 | 0 | 1.240 | 0.193 | 0.243 | 0.612 | 1.36 |
| soft0p9_fast | 2/3 | 0 | 1.211 | 0.162 | 0.254 | 0.613 | 2.66 |
| nominal1p0_fixed | 3/3 | 0 | 1.283 | 0.088 | 0.247 | 0.618 | 1.84 |
| nominal1p0_fast | 3/3 | 0 | 1.246 | 0.090 | 0.256 | 0.619 | 1.86 |
| stiff1p2_fixed | 1/3 | 0 | 1.310 | 0.125 | 0.256 | 0.139 | 2.22 |
| stiff1p2_fast | 0/3 | 0 | 1.270 | 0.178 | 0.243 | 0.138 | 3.14 |

结论：未通过。

该矩阵验证 PD 参数响应幅值，不等价于显式通信延迟或传感噪声；后二者仍需独立门禁。
