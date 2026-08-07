# Stage257：官方 X2 MuJoCo 执行器参数 × 上肢扰动直行门禁

冻结 actor、步态模板和门槛；PD Kp/Kd 同比扰动，并交叉固定/快摆臂。

| condition | full gate | fallback | forward median | lateral abs max | tilt max | stop z min | stop settle median |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft0p9_fixed | 3/3 | 0 | 1.234 | 0.184 | 0.243 | 0.632 | 1.84 |
| soft0p9_fast | 0/3 | 0 | 1.201 | 0.209 | 0.261 | 0.102 | 2.92 |
| nominal1p0_fixed | 3/3 | 0 | 1.298 | 0.098 | 0.248 | 0.638 | 1.60 |
| nominal1p0_fast | 3/3 | 0 | 1.247 | 0.148 | 0.263 | 0.628 | 2.02 |
| stiff1p2_fixed | 3/3 | 0 | 1.303 | 0.109 | 0.252 | 0.641 | 0.60 |
| stiff1p2_fast | 2/3 | 0 | 1.275 | 0.111 | 0.257 | 0.636 | 0.52 |

结论：未通过。

该矩阵验证 PD 参数响应幅值，不等价于显式通信延迟或传感噪声；后二者仍需独立门禁。
