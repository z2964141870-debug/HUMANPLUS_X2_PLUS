# 第七轮 D 预注册：生存时长删失诊断

日期：2026-07-28  
状态：待执行；仅诊断，不改变 Stage7C 裁决

## 背景

Stage7C 按预注册门失败：`NEUTRAL05` 的聚合 lateral max 比 BASE 高约 4.6%。
但分层后发现：

- 双方都生存的 14 个配对案例，候选 lateral 平均差为 `-0.0044 m`；
- 双方都失败的 18 个案例，候选 lateral 平均差为 `-0.0079 m`；
- lateral 聚合退化主要来自 3 个 `BASE 失败 → 候选生存` 的案例：候选平均
  多运行约 105 steps 后继续积累横移；
- 同时存在 1 个 `BASE 生存 → 候选失败` 的反例。

直接比较不同生存时长的最大横漂会混入删失偏差。本轮记录逐步轨迹，并只在
每对案例共同存活的窗口内比较。

## 假设

若 Stage7C 的 lateral 失败主要是时长删失造成，则在
`min(BASE steps, NEUTRAL steps)` 的严格匹配窗口内，候选的 lateral max/AUC
应不劣于 BASE；若匹配窗口内仍明显更差，则“救活但侧漂”的解释成立。

## 干预与对照

- 完全复用 Stage7C 的 BASE/NEUTRAL05、seed17/42/73、12-case、扰动和
  400-step 配置；
- 唯一变化是记录每个有效 step 的 lateral、世界 heading、tilt、root-z；
- 逐 case 使用共同有效步数，不填充倒地后的虚假状态；
- 不训练、不改 checkpoint、不改 reward、不改 Stage7C 门槛。

## 诊断判据

以下全部满足，只能得出“局部信号值得下一轮受约束训练”，不能晋级模型：

1. 36 对匹配窗口的平均 lateral max 不高于 BASE；
2. 36 对匹配窗口的平均 lateral mean/AUC 不高于 BASE；
3. 世界 heading 的匹配窗口均值不高于 BASE；
4. 14 个双方生存案例中无 lateral `+0.15 m` 回归；
5. 三个被救活案例在 BASE 失效时刻之前，候选 lateral 不出现 `+0.15 m`；
6. 记录轨迹的复跑与原 Stage7C 汇总指标逐项可重复。

若通过，下一轮只能针对“长期 lateral 锚定”做最小受约束实验；若不通过，
停止沿当前 Future Adapter 继续训练。

