# X2 速度保底线与 WBT 晋升线 Phase 2 裁决

日期：2026-08-09
状态：**BASE 技能路由保留为架构能力但未晋升；WBT 联合 reference repair 被官方 free-root 物理门否决。**

## 一句话结论

本轮把 BASE 的不稳定停车进一步定位到 brake/contact/momentum 在 recovery 交权前发生分叉，并证明双技能路由可把移动门做到 `5/5`，但同语义合并结果仅 `5/8`，停车可靠性不足；WBT 联合修复虽把 fixed-root 腿 RMSE 从 `0.1583` 降到 `0.1406 rad`、free-root 滑移从 `0.0739` 降到 `0.0671 m/s`，却使生存比例从 `0.236` 降到 `0.199`，因此长训与 WBT PPO 继续锁定。

## 主线 A：BASE 参数保护与技能交权

### 假设

1. Stage337-f010 的短暂 `4/5` 可能来自 recovery 收益，但继续训练的退化来自 actor 参数漂移。
2. 正常站立与停车恢复是不同吸引域，使用两个独立 actor slot 可能减少职责冲突。
3. 若失败只是交权太晚，倾角紧急 latch 应能救回临界样本。

### 干预 / 对照

- 参数保护：从同一 source、seed 和 recovery reset 设置分叉，只增加 `actor_anchor_coeff=1.0`。
- 技能分离：冻结 source stand 负责开局；Stage337-f010 仅负责 brake 后 recovery；moving actor、PD、Future-intent 和门禁不变。
- 紧急交权：只在 `t≥0.8 s、speed≤0.10 m/s、tilt≥0.15 rad` 时允许单向提前切入 recovery；功能默认关闭。
- 所有最终裁决都来自官方 AimDK v1 MuJoCo 完整 `stand→start→move→brake→stop` 事件，不用训练 reward 晋升。

### 结果

| 实验 | 关键结果 | 裁决 |
|---|---|---|
| Stage341 actor anchor | 参数 RMS 漂移减少约 40%，但 full gate `0/5` | 淘汰，不扫系数 |
| Stage343 stand/recovery 分离 | 单轮 full `4/5`、move `5/5`、stop `4/5` | 保留接口，不晋升 |
| 相同双技能语义合并 | `5/8` | 说明停车收益不可靠 |
| Stage344 emergency latch | full `1/5`；2次触发均未救回 | 淘汰，默认关闭 |

### 结论

- 参数距离并不能代理闭环能力；全局 actor anchor 抑制了 recovery 所需的新行为。
- 双技能路由对初始站立和移动航向有结构性价值，但没有可靠改善停车成功率。
- 失败在 recovery latch 前的 brake 阶段已经出现倾角、接触和动量分叉；“检测到快倒再切 recovery”太晚。
- 下一次 BASE 训练若继续，应把 `move→decelerate→contact event→stop` 作为单个训练 episode 的事件，而不是继续增加静态 recovery reset、交权阈值或短训分支。

## 主线 B：WBT 联合 reference repair oracle

### 假设

foot-only 修正失败可能是因为 root/COM、接触时序、支撑脚锁定与摆脚 terminal 没有联合处理；若只是低维耦合缺失，一个有界联合 oracle 应在官方物理中同时改善跟踪、滑移和 free-root 生存。

### 干预 / 对照

- 在 current-v4 exact30 上搜索有界 root 支撑偏移/高度、`±2` 帧相位、stance-foot 小残差 IK、`0–2 cm` 摆脚 clearance 与 `0–25%` terminal blend。
- 先过离线连续性/几何门，再进入官方 AimDK v1 的 prescribed-root 与 free-root 两种物理测试。
- oracle 使用完整未来和物理结果选参，只用于存在性验证，不能作为可部署 adapter。
- COM、DCM、contact 都是 MuJoCo 模型估计，不是真实足底力、COP 或实机动力学真值。

### 结果

| variant | fixed-root 腿 RMSE(rad) | free-root 生存比例 | slip p95(m/s) | free-root 全程 |
|---|---:|---:|---:|---:|
| current-v4 | 0.1583 | 0.236 | 0.0739 | 0/5 |
| joint repair oracle | 0.1406 | 0.199 | 0.0671 | 0/5 |

- 5类动作每类均有 `9–12/12` 个候选通过离线门，说明不是实现完全失效。
- fixed-root 关节跟踪和总体滑移有局部改善，但 4/5 动作生存更差；只有 `walk_to_stand` 从 `0.347` 微升至 `0.351`。
- 这证明局部几何/接触指标可被优化，却没有形成可持续的 free-root 平衡闭环。

### 结论

- 当前低维联合修复不足以把 reference 变成动力学可行的 X2 Silver 数据。
- 停止 foot offset、时间平滑和小幅 root 参数扫描；它们的信息增益已经很低。
- 下一条 WBT 有意义的路线是显式接触约束轨迹优化或 X2 原生 dynamic teacher，之后仍必须回到同一官方 free-root 门禁。

## 最终晋升状态

- `BASE_LOCOMOTION`：保持冻结。
- `BASE_TRANSITION`：保持冻结；Stage343 仅保留为双技能接口证据。
- recovery 研究 checkpoint：仍保留 Stage337-f010/model_155，不宣称可靠晋升。
- Stage341 anchor：负结果，不作为候选。
- emergency latch：默认关闭。
- WBT dynamic Silver：仍为 `0`；Any2Any 式 WBT PPO 与长训继续锁定。
- 真机：未发送任何命令，未升级 SDK/固件。

## 下一阶段的唯一高价值工作

1. BASE：构造并训练完整的减速—接触—停车事件课程，先用官方 5-run 门判定是否值得扩到固定矩阵。
2. WBT：用显式接触约束轨迹优化或 X2 原生 rollout 产生 dynamic teacher；没有通过 free-root 门前不启动 PPO。
3. 不再投入：actor anchor 系数、emergency 阈值、foot-only/平滑/root 小参数扫描。

## 证据

- BASE Phase2：[x2_recovery_phase2_skill_routing.md](baseline/x2_recovery_phase2_skill_routing.md)
- WBT Phase2：[x2_joint_reference_repair_phase2.md](retarget/x2_joint_reference_repair_phase2.md)
- anchor 门禁：[stage342_recovery_f010_anchor1_u5_gate.json](official_x2/stage342_recovery_f010_anchor1_u5_gate.json)
- 双技能门禁：[stage343_dual_stand_recovery_f010_u5_gate.json](official_x2/stage343_dual_stand_recovery_f010_u5_gate.json)
- emergency 门禁：[stage344_dual_recovery_emergency_latch_gate.json](official_x2/stage344_dual_recovery_emergency_latch_gate.json)
