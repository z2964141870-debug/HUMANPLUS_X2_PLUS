# Stage251：官方 X2 MuJoCo 上肢扰动直行门禁

同一冻结后端、同一行走/停车配置；只改变 14DOF 上肢目标。上肢轨迹来自已有 X2 重定向真实动作，不是临时正弦动作。

| profile | full gate | fallback steps | forward median (m) | lateral median (m) | tilt max median (rad) | stop settle median (s) | upper RMSE (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 3/3 | 0 | 1.196 | -0.099 | 0.249 | 1.96 | nan |
| slow | 3/3 | 0 | 1.218 | -0.120 | 0.250 | 2.10 | 0.038 |
| fast | 3/3 | 0 | 1.201 | -0.152 | 0.263 | 1.88 | 0.040 |

结论：通过。

此门禁只证明直行条件下的扰动鲁棒性；转向与执行器参数扰动仍需单独验证。
