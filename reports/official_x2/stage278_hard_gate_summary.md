# Stage278：官方 X2 最难复合门禁

固定评估门槛；条件为 soft PD 0.9、快速摆臂、0.30 m/s 直行及 1 秒 brake→stand 平滑交权。

| run | full | move | stop | heading max | stop drift | stop z min | settle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | False | False | True | 0.3124 | 0.1404 | 0.6167 | 3.42 |
| 2 | True | True | True | 0.2998 | 0.1349 | 0.6154 | 2.90 |
| 3 | True | True | True | 0.2974 | 0.1453 | 0.6147 | 1.86 |

- 复合停车门：3/3，已闭合。
- 完整门：2/3，尚未闭合。
- 最坏航向：0.3124 rad（门槛 0.3000 rad）。

结论：平滑 brake→stand 已把原先的停车倒地稳定修复；剩余瓶颈是快速摆臂下的航向裕量，不能据此解锁长训。
