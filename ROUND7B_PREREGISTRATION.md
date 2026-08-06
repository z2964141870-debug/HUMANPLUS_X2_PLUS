# 第七轮 B 预注册：Conservative Adapter + Heading Risk

日期：2026-07-28  
状态：已由 Phase A 解锁；待 5/25 级联

## 解锁证据

Phase A 仅按 selection 锁定 `alpha=0.75`，它虽在三个训练分布动作上将平均
heading/lateral 改善约 43%/51%，却使留出 wave-left 的 heading 比 BASE
恶化 `0.160 rad`、tilt 达 `0.531 rad`。

较小的 0.25/0.50 显著减轻该回归，且 failure mode 明确集中在
heading/tilt，满足 Phase B 解锁条件。

## 假设

Stage6 Adapter 的局部正信号被训练目标中的世界航向约束缺失所污染。保持
residual 权限减半，并在训练时显式惩罚 heading-target 环境的累计世界航向
误差，应能优于同预算中性训练。

## 干预与对照

两支均从 Stage208-s2550 独立开始：

1. `NEUTRAL05`：FUTURE + gait phase，coordination blend `0.50`，
   heading error weight `0.00`；
2. `RISK05`：完全相同，但 heading error weight 固定为 `0.25`。

共同合同：

- actor 主体冻结；
- residual 只拥有原 8 个腰髋协调模态；
- 最终动作 residual 最大 `0.05`；
- 训练上肢、速度范围、75% ideal + 25% nominal+delay、seed42 与
  Stage6 完全相同；
- 不新增 lateral reward、网络、horizon、数据或执行器参数；
- 5 和 25 updates 都从 Stage208 独立训练，不串联 checkpoint；
- 每支只保存 final checkpoint。

## Gate5

两支均先过 finite/reload/frozen/mask/residual 结构门，再运行同一个
12-case nominal+delay 面板。只有 RISK05：

- 生存数和总生存步数不低于 NEUTRAL05；
- 平均 heading 不高于 NEUTRAL05；
- 平均 lateral 不高于 NEUTRAL05；
- BASE 的正常速度生存案例全部保留；
- 正常速度 wave-left heading 不超过 BASE+0.15 且 tilt≤0.45；
- 相对 NEUTRAL05 至少改善一个留出案例的 heading 或生存；

才解锁 25 updates。训练 reward/episode length 不参与解锁。

## Gate25

RISK05 必须同时满足：

- 生存数不低于 BASE 和 NEUTRAL05；
- 总生存步数不低于 BASE 和 NEUTRAL05；
- 低速总步数不低于 BASE；
- 全面板平均 heading/lateral 均优于 BASE 与 NEUTRAL05；
- BASE 正常速度生存案例全部保留；
- 所有正常速度留出动作不发生 heading/lateral `+0.15` 或生存 tilt
  `>0.45` 的回归；
- residual `≤0.050001`。

否则保留 Stage208；无论训练曲线多好，都禁止继续 200+ updates。

## 诚实边界

- 这是风险约束的因果验证，不是完整 SONIC；
- world heading 只用作训练 reward，未增加部署 observation；
- 单 seed、CPU compatibility mode 的结果仍不能直接上真机；
- 若只在正常速度通过，最多形成有限运行 envelope。
