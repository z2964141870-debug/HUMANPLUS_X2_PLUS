# X2 速度保底线与 WBT 晋升线 Phase 1 裁决

日期：2026-08-09
状态：**BASE 获得可复现改善信号但未过晋级门；WBT 排除 foot-only 修补，PPO 继续锁定。**

## 一句话结论

独立 recovery reset 课程把 stiff-fixed 全门通过率从旧后端的 `3/5` 提高到最佳中间 checkpoint 的 `4/5`，证明方向有价值，但继续 5 个 update 后退化为 `1/5`；WBT 的 toe→forefoot 修补虽改善固定-root跟踪，却使 free-root 生存和滑移显著恶化，因此两条线都还不能进入长训或替换冻结资产。

## 主线 A：stop-to-stand / recovery

### 假设

stand backend 的主要缺口之一，是训练初始化没有覆盖真实停车末段的临界与失败状态；以低比例注入这些物理状态，应当扩大恢复吸引域，而不破坏原有起步和直行。

### 干预

- 从 Stage335 的 90 个官方 MuJoCo 停车边界状态中，仅恢复 root pose/velocity 与 31DOF `q/dq`。
- 课程只接入独立 `stand_backend`，并位于原 reset 之后；`fraction=0` 严格 no-op。
- 从同一个 stand `model_150.pt`、同 seed 47 分叉，比较 `fraction=0/0.05/0.10`，每支只做 5 个 PPO update。
- 仅对最佳 `0.10` 分支继续 5 个 update，检验改善能否随训练延续。

### 对照

- 移动策略、部署 supervisor、Future-intent、PD、upper=fixed、速度与官方 AimDK v1 MuJoCo 均固定。
- 历史 source stand backend 与每个新候选都使用完整 `stand→start→move→brake→stop` 门禁。
- 最终统一为每支 5 次有效重复；训练 reward 不参与晋升判定。

### 结果

| stationary backend | 训练干预 | official stiff-fixed full gate | 裁决 |
|---|---|---:|---|
| source stand backend | 无 recovery 课程 | 3/5 | 冻结基线 |
| Stage337 f000 | 0%，5 update | 1/5 | 淘汰 |
| Stage337 f005 | 5%，5 update | 1/5 | 淘汰 |
| Stage337 f010 | 10%，5 update | **4/5** | 保留研究候选，未晋升 |
| Stage339 f010 | 同分支继续至10 update | 1/5 | 淘汰，不再续训 |

Stage337-f010 的唯一失败没有倒地，但行走航向超过门限；Stage339 则重新出现两次停车倒地。也就是说，低比例 recovery 确实改善了恢复状态覆盖，但继续 PPO 已开始破坏移动/航向闭环。

### 结论

- recovery reset 的**接口与因果方向成立**：最佳点从 `3/5` 提到 `4/5`。
- 它尚未满足任务卡的 stiff-fixed `5/5`，不能替换 `BASE_TRANSITION` 或解锁长训。
- “更多训练自然会更好”在这个分支上被直接证伪：5→10 update 从 `4/5` 退化为 `1/5`。
- 当前甜点位是 Stage337-f010/model_155；保留用于后续机制研究，但不称为可靠 checkpoint。

### 下一步

不再延长 Stage339，也不再扫 recovery fraction。下一项最有信息量的单变量，是在 Stage337-f010 的 5-update 训练中加入固定 actor anchor / trust-region，约束 stand 分支不要改变影响移动航向的已有动作；仍只做极短训练并回到同一 5-run 官方门禁。

## 主线 B：官方 X2 Whole-Body Tracking reference

### 假设

早期动态数据可能同时含有 wrist 终端语义错位、foot landmark 错位和时间分辨率不足；若它们是主因，修正后应同时改善 FK、连续性、接触和官方 free-root 物理表现。

### 干预与对照

- wrist：只把终端从 `wrist_yaw_link` 改为官方观测契约的 `wrist_roll_link`。
- foot：比较 current-v4 与语义更匹配的 toe→forefoot；依次只改变 60→30 解算、全关节 smoothing、knee-only smoothing。
- 物理筛查固定官方 scene/x2.xml、官方 29DOF PD、1 kHz physics / 50 Hz control；明确区分 prescribed-root trackability 与 free-root balance。

### 结果

- wrist-roll 对上肢与行走 smoke 的手腕误差局部改善约 15–20%，但全局最差腕 p95 仅 `0.2560→0.2519 m`，未过 5% 全局门；它只保留为上肢 specialist contract。
- 发现旧 Task3 的 9 条中有 1 条 KIT 100Hz 动作存在帧率契约错误：旧 266 帧，exact30 应为 240 帧。后续比较已改为 source-time 对齐。
- 60Hz IK 再降到30Hz不能消除 knee branch jump；smooth11/13/15 或 knee-only smoothing 虽降低 joint-step，却系统性放大 stance slip。
- toe→forefoot 在官方 prescribed-root 下，腿 RMSE `0.1583→0.1233 rad`（改善22.1%），饱和 `0.0080→0.0038`（减少52.9%）。
- 但在 free-root 下两组均 `0/5` 全程通过；toe 方案平均生存比例 `0.236→0.126`（下降46.7%），slip p95 `0.0739→0.1571 m/s`（约2.13倍）。

### 结论

foot/ankle 语义错配是真问题，但不是整体步态失败的充分解释。局部足部几何更好，反而会通过 root-contact 耦合更快失稳；继续输出端平滑或 foot-only offset 已进入低价值区。current-v4 继续作为对照，toe-smooth9 不晋升 Bronze/Silver，也不得进入 PPO。

### 下一步

reference 修复转向**联合约束**：root/COM 轨迹、接触相位、stance-foot 约束和 foot terminal 同时进入一个短窗口优化问题；先在5条诊断步态上要求 free-root 生存与滑移不劣化，再谈 Silver 与忠实 Any2Any。

## 资产与可复现证据

- BASE reset 审计：[x2_recovery_phase1_pretrain.md](baseline/x2_recovery_phase1_pretrain.md)
- 5-update 官方面板：[stage338_recovery_candidate_panel.json](official_x2/stage338_recovery_candidate_panel.json)
- 10-update 反证：[stage340_recovery_u10_gate.json](official_x2/stage340_recovery_u10_gate.json)
- wrist A/B：[x2_wrist_terminal_contract_ab.md](retarget/x2_wrist_terminal_contract_ab.md)
- timebase/smoothing：[x2_forefoot_smoothing_ab.md](retarget/x2_forefoot_smoothing_ab.md)
- knee-only A/B：[x2_forefoot_knee_continuity_ab.md](retarget/x2_forefoot_knee_continuity_ab.md)
- 官方物理筛查：[x2_forefoot_official_physics_screen.md](retarget/x2_forefoot_official_physics_screen.md)

## 最终门禁状态

- `BASE_LOCOMOTION`：保持冻结，不变。
- `BASE_TRANSITION`：保持冻结，不被 Stage337/339 替换。
- recovery 研究候选：Stage337-f010/model_155 保留；未晋升。
- Stage337-f010 checkpoint / ONNX SHA-256：`b29ee2a3…4235a` / `bd2a1f19…54f2a`。
- Stage339 反证 checkpoint / ONNX SHA-256：`1d8c8e62…f63158` / `6548c94f…bb6dd`；只保留为负结果，不作为候选。
- WBT dynamic Silver：仍为 0；忠实 Any2Any PPO 仍锁定。
- 真机：未发送任何命令，未升级 SDK/固件。
