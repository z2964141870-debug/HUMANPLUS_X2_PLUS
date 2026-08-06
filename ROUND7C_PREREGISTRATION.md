# 第七轮 C 预注册：成对初态扰动鲁棒性复测

日期：2026-07-28  
状态：待执行

## 背景

`RISK05` 未通过 Gate5，禁止继续训练；同源的 `NEUTRAL05` 却在固定初态
12-case 面板上形成意外 Pareto 候选：生存不降，总步数、低速步数、
heading 与 lateral 均优于 Stage208 BASE。

固定初态只能说明候选值得复测，不能证明它真正提高了闭环鲁棒性。本轮不再
训练，只判断该收益能否跨初态扰动和随机种子复现。

## 假设

若 `NEUTRAL05` 学到的是可泛化的上肢—步态协调，而不是对固定初态/固定轨迹
的偶然拟合，那么在相同初态姿态和速度扰动下，它应继续不劣于 BASE。

## 干预与对照

- 对照：Stage208 BASE，adapter disabled；
- 干预：`NEUTRAL05` Gate5 final，future+phase，blend `0.50`；
- 两者使用相同 12-case、nominal+delay、400 steps；
- 随机种子固定为 `17 / 42 / 73`；
- 每个种子施加：
  - 初始 roll、pitch：各 `[-0.02, +0.02] rad`；
  - 初始 yaw：`[-0.05, +0.05] rad`；
  - 初始世界 lateral velocity：`[-0.05, +0.05] m/s`；
  - 初始 yaw rate：`[-0.05, +0.05] rad/s`；
- 每个 BASE/NEUTRAL 对使用相同 seed 与同一配置；
- 不修改 checkpoint、reward、网络、动作幅值、执行器或 reference。

## 配对完整性门

每个 seed 下，两支的 12 个 case 必须逐项拥有相同的：

- 初始 root quaternion/yaw；
- 初始 root linear/angular velocity；
- 初始 base tilt。

任一不一致，本轮不可解释，须先修评估器。

## 通过门

聚合 36 个 case，`NEUTRAL05` 必须同时满足：

1. 生存数不低于 BASE；
2. 总生存步数不低于 BASE；
3. 低速总步数不低于 BASE；
4. 平均绝对世界 heading error 不高于 BASE；
5. 平均 lateral drift 不高于 BASE；
6. BASE 的正常速度生存 case 不丢失；
7. 正常速度留出动作不出现 heading/lateral `+0.15`，且生存案例
   tilt 不超过 `0.45 rad`；
8. 至少 2/3 个 seed 同时做到总步数、世界 heading、lateral 不劣于 BASE；
9. residual 最大值不超过 `0.050001`。

## 裁决

- 全部通过：`NEUTRAL05` 晋级为“扰动鲁棒候选”，但不宣称迁移完成；
- 仅单 seed 或固定初态通过：判为不稳健，不替代 Stage208；
- 配对失败：只修评估器，不训练；
- 本轮结束后不根据结果临时改门槛。

## 诚实边界

本轮仍是单一 nominal+delay 模型、CPU compatibility mode、仿真评估。通过只
说明迁移工作出现了可复现的局部闭环收益，不代表完整 SONIC、真实 X2 或
任意动作已经解决。
