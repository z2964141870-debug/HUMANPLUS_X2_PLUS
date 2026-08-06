# 第七轮预注册：Conservative Intervention / Residual Pareto

日期：2026-07-28  
状态：Phase A 待运行；风险约束训练锁定

## 最终命题

Stage6 FUTURE-s2575 已学到部分正常速度动作的方向协调，但 full-strength
residual 破坏了留出 wave 且未改善低速。第七轮先检验：

> Stage6 的失败是否主要来自介入强度过大，而不是 future-intent 映射本身无效。

## Phase A：零训练 residual blend

冻结同一个 FUTURE-s2575 checkpoint，不改 actor、critic、上肢、reference、
执行器、reward 或 gait template，只将最终腰髋 residual 乘以：

```text
alpha ∈ {0.25, 0.50, 0.75}
```

现有 `BASE` 是 alpha=0，Stage6 FUTURE 是 alpha=1。缩放发生在已经经过
intent gate 和 `±0.10` 硬界之后，因此 alpha=0 必须严格回退 Stage208，
alpha≤1 不会扩大训练分布外的动作权限。

## 选择集与留出集

alpha 的选择只允许读取正常速度 `0.30 m/s` 的三个训练分布动作：

1. swing arms；
2. knocking；
3. box lift。

锁定 alpha 后才裁决：

- 真实采集 wave；
- KIT wave-left；
- Female lift-box；
- 全部 `0.20 m/s` 低速条件。

虽然批量仿真一次产生所有条件，分析器必须分别标记 selection 与 held-out；
不得根据留出结果换 alpha。

## Phase A 选择门

在三个 selection 动作上，候选 alpha 必须：

- 3/3 生存满 400 步；
- 每条 tilt `≤0.45 rad`；
- 每条 residual `≤0.100001×alpha`；
- 平均 heading 和 lateral 都比 BASE 至少改善 10%。

满足门的 alpha 中，按 selection 的归一化
`heading/base_heading + lateral/base_lateral` 最小者锁定；平分时取更小
alpha。

## 留出晋级门

锁定 alpha 只有同时满足以下条件才形成 Pareto：

- 总生存数不低于 BASE 的 5/12；
- BASE 已生存的正常速度案例全部保留；
- 正常速度留出案例 heading/lateral 不比 BASE 恶化 0.15；
- 正常速度生存案例 tilt `≤0.45 rad`；
- Female lift 的生存步数不低于 BASE；
- 六个低速案例总生存步数不低于 BASE 的 962；
- 全面板平均 heading 和 lateral 均优于 BASE。

## 预注册的速度 fallback

若锁定 alpha 只失败于低速总生存，而正常速度全部通过，则允许一个不训练、
可部署的 command gate：

```text
vx ≤ 0.20 m/s  → alpha=0（Stage208 fallback）
vx ≥ 0.30 m/s  → locked alpha
中间线性插值
```

该结果只能称为“正常步速安全工作区”，不能声称解决低速 locomotion。

## Phase B 解锁条件

只有以下情况才允许另写 addendum 并训练：

- selection 三动作的平均 heading/lateral 至少改善 15%；
- held-out 的主要失败明确是 heading/tilt，而非生存普遍崩塌；
- 至少一个 alpha 比 full-strength 明显减轻 held-out 回归。

若解锁，下一干预优先是显式 world-heading 风险奖励与保守 residual，而不是
扩大网络、horizon 或 residual 上限。训练前必须补写权重、对照和停止门。

## 停止条件

- 没有 alpha 通过 selection 门：Stage6 future-intent 支线停止；
- selection 通过但所有 alpha 都伤正常速度 held-out：判为动作分布过拟合；
- 只有 command gate 通过：保留有限工作区，不接真实 SONIC；
- 单 seed CPU compatibility mode 的正结果仍需真正随机多 seed 才能晋级。
