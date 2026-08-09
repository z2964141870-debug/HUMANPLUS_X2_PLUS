# X2 速度保底线门禁覆盖与缺口

日期：2026-08-09
依据：`x2_speed_backend_manifest.{md,json}` 与现有官方 MuJoCo 报告。

## 计数规则

- 最终任务卡矩阵是 `PD 3 × upper 3 × motion 3 × repeat 3 = 81` 条完整事件 rollout；每条必须覆盖 stand/start/cruise/brake/stop。
- 只有同一 checkpoint、同一部署合同、同一门槛下运行的面板才能报告面板通过率。
- 下表保留“分散覆盖”，但不把不同 Stage、supervisor、yaw gain 或 preview 配置的结果相加为一个候选的完整通过率。
- `full` 表示 startup/move/stop 全门通过；不能用 survival、单一 stop pass 或基础设施失败替代。

## BASE_LOCOMOTION 分散覆盖

| 面板 | 权重 | PD | upper | motion | repeats | 结果 | 是否完整面板 |
|---|---|---|---|---|---:|---:|---|
| Stage250 | Stage219-s2600 | nominal | fixed | low/medium straight/right/left | 24 | 24/24 | 否：混合两个速度，但无 upper/PD 交叉 |
| Stage251 | 同上 | nominal | fixed/slow/fast | straight | 9 | 9/9 | 否：无 turn、无非 nominal |
| Stage252 | 同上 | nominal | fixed/slow/fast | right/left | 18 | 17/18 | 否，且 fast-left 失败 |
| Stage253 | 同上 | nominal | fixed/slow/fast | left only，yaw gain=2.5 | 9 | 9/9 | 否：只重跑左转，不能与 Stage252 静默合并 |
| Stage257 | 同上 | 0.9/1.0/1.2 | fixed/fast | straight | 18 | 14/18 | 否：缺 slow/turn |
| Stage263 | 同上，另一 supervisor | 0.9/1.0/1.2 | fixed/fast | straight | 18 | 14/18 | 否；与 Stage257 失败位置不同但不能拼接 |

证据：

- `reports/official_x2/stage250_rsl_action_contract_fix_20260808.json`
- `reports/official_x2/stage251_upper_straight_panel.json`
- `reports/official_x2/stage252_upper_turn_panel.json`
- `reports/official_x2/stage253_left_yaw_gain_panel.json`
- `reports/official_x2/stage257_pd_upper_straight_panel.json`
- `reports/official_x2/stage263_pd_upper_straight_panel.json`

## BASE_TRANSITION 分散覆盖

| 面板 | 权重 | 部署配置 | PD | upper | motion | repeats | 结果 |
|---|---|---|---|---|---|---:|---:|
| Stage314 | Stage306-s2652 | split-intent | nominal | fixed/fast | straight | 6 | 6/6 |
| Stage321 | 同上 | preview=0.5 s；right gain=3.0 / left=2.5 | nominal | fixed/fast | right/left | 8 | 7/8 |
| Stage325 | 同上 | preview=0.5 s；lateral supervisor | nominal | fixed/fast | straight | 6 | 6/6 |
| Stage326 | 同上 | preview=0.5 s；split-intent brake | stiff 1.2 | fixed/fast | straight | 10 | 8/10（fixed 3/5，fast 5/5） |

证据：

- `reports/official_x2/OFFICIAL_X2_TRANSITION_HEAD_STOP_INTENT_20260808.md`
- `reports/official_x2/stage321_future_stop_preview0p5_panel.json`
- `reports/official_x2/stage325_straight_preview_regression.json`
- `reports/official_x2/OFFICIAL_X2_STAGE316_328_FUTURE_PREVIEW_AND_STIFF_AUDIT_20260808.md`

## 完整 81-case 目标的硬缺口

### BASE_LOCOMOTION

- [ ] 同一 runner 中重跑 nominal fixed/slow/fast × straight/right/left；现有分散证据不能作为原子 27/27。
- [ ] soft 0.9 × slow upper × straight/right/left。
- [ ] soft 0.9 × fixed/fast × right/left。
- [ ] stiff 1.2 × slow upper × straight/right/left。
- [ ] stiff 1.2 × fixed/fast × right/left。
- [ ] 修复或训练后让已测 straight 复合矩阵从 14/18 提升到 18/18。

### BASE_TRANSITION

- [ ] nominal slow upper 的 straight/right/left。
- [ ] nominal fast-left 的稳定重复通过（当前 1/2）。
- [ ] soft 0.9 的全部 upper/motion。
- [ ] stiff 1.2 slow upper 的全部 motion。
- [ ] stiff 1.2 fixed/fast 的 right/left。
- [ ] stiff-fixed straight 从 3/5 提升并独立复验至少 5/5。
- [ ] 一次冻结配置的完整 81-case 原子运行。

### 两个候选共同未覆盖

- [ ] 显式通信 delay。
- [ ] observation delay/lag 的官方 ROS 闭环，而不是 direct-MJCF 诊断。
- [ ] sensor/action noise。
- [ ] foot slip、contact timing、swing clearance 的正式统一报告字段。
- [ ] action saturation、action delta、jerk、handoff delta 的完整矩阵统计。
- [ ] 真机接口与真实硬件验证。

## 不计入晋级的结果

- Stage224/225 direct-MJCF delay/noise 探索：当前是未跟踪诊断文件，闭环语义与冻结官方 runner 不同；只可用于假设生成。
- Stage329/331：zero-upper checkpoint 早期小样本 3/3，但独立复验 fixed 3/5、fast 4/5，未超过 Stage306。
- Stage332/333：upper-activity 条件化最高 2/3，未晋级。
- 任意仅“停车 3/3”但行走航向失败的 Stage278 类结果：不等于 full gate。

## 结论

当前最可信事实是：`BASE_LOCOMOTION` 提供 nominal 安全回退，`BASE_TRANSITION` 在 nominal 起停和部分转向上提供 Future-intent 增益；但任何一个都没有完成任务卡要求的统一矩阵。最近失败已稳定收敛到 `stiff + fixed upper + stop-to-stand/recovery` 与 fast-left 航向裕度，下一阶段应先解决独立恢复吸引域，再重跑同候选完整矩阵。
