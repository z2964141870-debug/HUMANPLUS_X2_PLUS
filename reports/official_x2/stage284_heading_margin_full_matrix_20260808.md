# Stage284：快速上肢航向续训完整矩阵裁决

## 结论

Stage281-s2623 在官方 AimDK X2 MuJoCo 的 `PD×上肢` 18 条矩阵中通过 `12/18`，不能晋级；它把 soft/nominal 域改善到 `11/12`，但 stiff `1.2×PD` 仅通过 `1/6`。这证明快速上肢与显式航向续训有效，但当前 Future-intent adapter 没有执行器响应条件，不能解决 PD-only 变化。

## 干预

- 基底：Stage267-s2620；Stage219 actor 继续逐位冻结。
- 仅续训3个 PPO update，学习率 `1e-4`，desired KL `0.003`。
- 上肢速度从训练时 `0.20 rad/s` 对齐到官方 hard gate 的 `0.40 rad/s`。
- 航向误差权重从 `0` 调到 `0.5`。
- 固定 Stage278 的 1秒 `brake→stand` 平滑停车契约；评估门槛不变。

## 结果

| 执行器域 | 固定上肢 | 快速上肢 | 合计 |
|---|---:|---:|---:|
| soft 0.9 | 3/3 | 2/3 | 5/6 |
| nominal 1.0 | 3/3 | 3/3 | 6/6 |
| stiff 1.2 | 1/3 | 0/3 | 1/6 |
| **总计** | **7/9** | **5/9** | **12/18** |

机器可读结果：`stage284_pd_upper_straight_panel.json`。

## 可证实判断

1. 第3 update 是真实早停峰值：单独 hard-gate 三次均通过，航向峰值约 `0.281/0.293/0.290 rad`；第1和第5 update 更差。
2. 完整矩阵扩大重复后，soft-fast 仍有一次 `0.3102 rad` 航向越界，说明裕量不足而非完全解决。
3. nominal 域6/6全过，说明续训没有破坏标称执行器域。
4. stiff 域的主要问题同时包含正向 yaw 漂移和停车掉出 stand 吸引域。
5. 固定上肢时 upper intent 为零，当前 adapter 被硬门控为零；因此随机化 PD 并不会让它获得 PD-only 适配能力。

## 下一步

保留 Future-intent branch 处理上肢扰动，新增独立的低维 response-conditioned branch：只读取可部署的 lower q/dq/last-action 与 gait phase，只输出已有8个髋腰协调模式，并零初始化、冻结原 Future-intent branch。它必须先证明 stiff-fixed 和 stiff-fast 改善，才能重新跑18条矩阵。
