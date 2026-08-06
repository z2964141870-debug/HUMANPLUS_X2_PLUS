# 第七轮结果卡：保守 Future-Intent、风险约束与多种子扰动

日期：2026-07-28  
状态：完成；当前目标停止，Stage208 继续作为正式基线

## 一句话结论

迁移并非完全无望：保守 Future-Intent 候选在 36 个真实随机初态扰动案例中把
生存从 `15→17`、总步数从 `9674→9961`、世界航向误差降低约 4.9%；但它同时
增加横向漂移、丢失一个基线可过案例，严格鲁棒门失败，因此不能晋级，也不应
沿同一训练目标继续堆轮数。

## 假设

Stage6 的 Future+phase Adapter 已出现局部方向信号，但 residual `0.10` 和缺少
世界方向风险约束可能导致留出动作回归。第七轮依次检验：

1. 减小 residual 是否能保留收益、压住回归；
2. 显式 heading 风险项是否优于同预算中性训练；
3. 意外出现的中性候选能否跨随机初态复现；
4. 横漂失败是否只是不同生存时长造成的删失偏差。

## 干预与对照

- Stage7A：冻结 FUTURE-s2575，训练外扫描 residual blend
  `0.25 / 0.50 / 0.75`；
- Stage7B：从 Stage208 独立训练 5 updates：
  - `NEUTRAL05`：future+phase、residual≤0.05；
  - `RISK05`：相同配置，额外 heading-error reward；
- Stage7C：BASE 与 NEUTRAL05 在 seed `17/42/73` 下施加完全配对的初始
  roll/pitch/yaw、横向速度和 yaw-rate 扰动；
- Stage7D：重跑相同 36 对案例并记录逐步轨迹，在共同生存窗口内比较；
- 全部使用相同 12-case、nominal+delay、400-step 面板；未修改旧工程。

## 结果

### A. 静态 residual blend

- `alpha=0` 与 BASE 的 12×100 冒烟面板在 349 个数值字段上最大差 `0.0`；
- 仅按 selection 锁定的 `alpha=0.75` 将三个训练分布动作的 heading/lateral
  改善约 43%/51%；
- 但留出 wave-left 的 heading 比 BASE 恶化 `0.160 rad`，tilt 达
  `0.531 rad`，且低速总步数下降；
- 结论：训练外缩放只能减轻、不能消除跨动作回归。

### B. Heading 风险训练

| 分支 | 生存 | 总步数 | 低速步数 | heading | lateral |
|---|---:|---:|---:|---:|---:|
| BASE | 5/12 | 3176 | 962 | 0.6567 | 0.5259 |
| NEUTRAL05 | 5/12 | 3248 | 974 | 0.6031 | 0.4996 |
| RISK05 | 5/12 | 3175 | 966 | 0.6482 | 0.5072 |

`RISK05` 未优于同预算中性对照，禁止解锁 25 updates。`NEUTRAL05` 在固定面板
形成意外 Pareto 候选，因此只解锁多种子鲁棒复测，不解锁长训。

### C. 三随机种子成对扰动

初始 quaternion、倾角、线速度、角速度逐值配对，最大差严格为 `0.0`。

| 分支 | 生存 | 总步数 | 低速步数 | 世界 heading | lateral | tilt |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 15/36 | 9674 | 2788 | 0.5988 | 0.4657 | 0.5465 |
| NEUTRAL05 | 17/36 | 9961 | 2841 | 0.5697 | 0.4872 | 0.5111 |

候选在生存、步数、航向和倾角上出现聚合改善，但 lateral 增加约 4.6%，
BASE 正常速度生存案例未全部保留，且 `0/3` seed 同时做到
steps/heading/lateral 三项不劣。正式裁决为 `KEEP_STAGE208_BASE`。

### D. 共同生存窗口诊断

逐步轨迹复跑与 Stage7C 所有汇总数值最大差 `0.0`。

| 36 对共同窗口指标 | BASE | NEUTRAL05 | 差值 |
|---|---:|---:|---:|
| lateral max 的 case 均值 (m) | 0.458858 | 0.456815 | -0.002043 |
| lateral 时间均值的 case 均值 (m) | 0.178080 | 0.180017 | +0.001937 |
| 世界 heading 时间均值 (rad) | 0.239104 | 0.237962 | -0.001142 |

删失偏差得到部分确认：三个被候选救活的案例，在 BASE 失效时刻之前都没有
`+0.15 m` 横漂回归；但 lateral 时间均值仍轻微变差，另有一个双方都生存的
wave 案例横漂恶化 `0.1513 m`，刚刚越过预注册门。Stage7D 因而仍失败。

## 结论

1. Future intent + phase 的信号是真实的：它跨三种随机初态提高了聚合生存和
   航向，不再只是固定 seed 偶然结果；
2. 它尚不是可靠迁移模型：收益会在动作和初态之间交换，尤其没有稳定的
   lateral 锚定；
3. 单独添加 heading reward 没有解决问题；继续同目标长训、扫 residual 或
   事后回退都缺少证据；
4. Stage208 仍是正式基线，NEUTRAL05 仅保留为互补研究候选，不接 SONIC、
   不真机、不称“迁移完成”。

## 对“还有没有希望”的诚实回答

有希望，但希望来自“基线与候选表现互补”，而不是当前候选已经接近可部署。
候选救回 3 个 BASE 失败案例，同时只新增 1 个失败，说明少量条件化协调确实
能改变闭环吸引域；下一条有证据支持的路线应是**提前预测风险的选择性介入 /
模态化 Adapter**，让安全案例回退到 BASE、只在预计获益的动作阶段启用
residual，而不是让同一个 residual 全程处理所有动作。

## 停止条件与后续边界

- 当前 Future-Adapter objective 到此停止，不做 25/200/1000 updates；
- 下一轮若启动，必须是独立预注册的 predictive gate / mixture 路线，并以
  BASE、NEUTRAL05、事后 supervisor 三者为对照；
- 在多种子 held-out 门通过前，不接真实 SONIC 上肢流、不部署。

关键证据：

- `reports/stage7_conservative/stage7_blend_decision.md`
- `reports/stage7_conservative/stage7b_gate5_decision.md`
- `reports/stage7_conservative/stage7c_robustness_decision.md`
- `reports/stage7_conservative/stage7d_matched_horizon_decision.md`

