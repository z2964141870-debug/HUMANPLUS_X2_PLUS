# X2 动态重定向多方法赛马总结（2026-08-11）

状态：**所有预注册实验已停止；0 RL/PPO；进入版本归档。**

## 1. 目标与统一合同

本轮只回答一个问题：现有 `PHUMA-LUNGE-R-001 / Phase30 1.46x` 的运动学参考，能否经过动力学可行化，生成可供后续 tracker 学习的 X2 reference。

统一物理口径：

- official X2 `scene.xml`，MuJoCo 3.3.7，官方 Euler；
- physics 1 kHz、control 50 Hz；
- official `motion_control.yaml` 逐关节 PD，raw torque motor 独立重放；
- body29 受控、head2 固定 nominal；
- 每脚只统计 12 个 `contype!=0` 的 active sole spheres；排除每脚一个不可碰撞 visual mesh；
- 优化器自己的 loss 不是成功证书，最终只认 raw official replay；
- 没有因结果不好而追加 seed、调阈值、扩预算或进入 RL。

## 2. 重要方法学纠正

### 2.1 旧 Phase3 不能作为 DSMS 否证

旧实现同时存在 reference velocity 全零、官方 PD 错配、head constraints 未注册、head bounds 过期和 Euler 被替换等问题。Phase3 应视为 `INVALID_CONTRACT_IMPLEMENTATION`。Phase3b 修正合同后才形成第一条有效 DSMS 证据。

### 2.2 “3–5 cm 穿地”主要是 auditor 污染

旧 helper 每脚返回 13 个 geom，其中一个是 `contype=0` visual mesh。把 mesh size 当 sphere 半径会制造约 `-25～-51 mm` 的假穿透。真实 active-12 初始量级约为毫米级。

### 2.3 `-0.5 mm` 不是 official soft-contact 的绝对物理门

既有成功的 native closed AimDK trace 中，active-sole contact distance 最小约 `-4.487 mm`，p01 为左/右 `-2.116/-2.491 mm`。因此本轮保留旧预注册 `-0.5 mm` 判定作 provenance，但后续物理解释采用同 scene 的 native 分布校准，不能为追求零压入而调控制。

### 2.4 Phase4/5/6/8 存在 actuator 尾部假设错误

官方 head actuator 是索引 `15/16`，不是最后两个；最后两个是右腕 pitch/roll。历史代码部分使用 `[:-2]`/`[-2:]`，因此 Phase4/5 搜索空间不满足声明的 body29/head2，方法裁决降级。Phase6/8 经 Phase10 零搜索 name-based 投影重放后，核心结果仍分别为 `0.683/0.775 s`，Phase8 的26次切换和21 ms最长离地也逐值保留；所以 reset/SBTO 机制信号保留，但后续必须换成显式按名字构建的 runner。

## 3. 结果总表

| 路线 | 计算成本 | 核心结果 | 裁决 |
|---|---:|---|---|
| Phase3b corrected DSMS | 443 s，300 IPOPT iter | max defect `.05422`；raw 同时失败于 contact/foot speed/flight/root accel | 拒绝当前原始 tracking-cost 短前缀 |
| Phase3c contact-aware DSMS | 516 s，300 iter | max defect `.001154`，比3b改善97.9%；多项接触指标改善，但右足速、flight、root accel/head仍失败 | 机制有效，非 teacher |
| Phase4 SBTO 初跑 | 40 s，480 rollouts | 搜索目标混入 visual mesh，且body29/head2映射错误 | 实验污染，只拒绝 candidate，不裁决 SBTO |
| Phase5 DDR rolling CEM | 32 s | 8门过7，但body29/head2搜索映射错误；active penetration未改善 | candidate拒绝，不评价忠实DDR |
| Phase6 reset projection + fixed bridge | SLSQP 30 iter + replay | t0 双脚真实接触；短窗除旧 penetration 门外全过；完整跌倒 `.575→.683 s` | **必要公共前置层，仍非充分解** |
| Phase8 reset-aware corrected SBTO | 91 s，480 rollouts | 完整跌倒 `.683→.775 s`；出现26次卸载尝试，但最长单支撑仅21 ms，属于 chatter | **当前最强正信号，candidate仍拒绝** |
| Phase9 reset + phase-aware DSMS | 1203 s，300 iter | max defect `.1667`；raw `.245 s` root-z倒地，早于 `.333 s` liftoff起点 | 当前长窗口 DSMS 配置失败 |
| Phase11 name-mapped event SBTO | 41.7 s，216 rollouts | 正确body29/head2；右支撑稳定，但左脚100% stuck、0 switch，fall `.702 s` | 固定4-mode/8参数事件表示停止 |
| Phase12 task-space/load preflight | 0 physics，1次FK/Jacobian | 8D任务Jacobian满秩；但固定双足把COM移至右足需最大4.943rad、2关节越限 | **物理前拒绝固定足位载荷转移表示** |
| Phase13 full support-margin audit | 0 physics，175次FK | 137/137单支撑帧COM投影在意图支撑面外，gap p50/p95=.214/.392m；足距p50=.801m | **原stance标签不能作X2硬接触真值** |
| Phase14 label-only contact repair | 0 physics，175次FK | 需改137/175帧，145帧仅能退为DS，仍有30帧连DS都不覆盖COM | **仅改contact标签失败，q/root/足位必须变** |
| Phase15 morphology normalization | 0 physics，确定性公式 | hip-roll scale=.342使足距p50 .801→.464m并修复DS覆盖；SS仍137/137失败、需root-z改10.4cm | **可作联合优化初值，不能作teacher** |

## 4. 当前最可信结论

1. **reset-compatible 初态投影必须成为所有方法的固定前置层。** 它以约 3 mm root-z、0.0125 rad 关节 RMS 的小改动修复 t0 双脚接触，并把确定性生存时间从 0.575 s 提高到 0.683 s。
2. **主要矛盾不是持续严重穿地，而是支撑转换失败。** Phase6 跌倒前双脚始终接触、无 flight、无 contact switch；左脚在非意图期仍 100% 接触，身体在低滑移双脚约束下倾倒。
3. **Phase8 sampling 是本轮唯一干净的方向性改善。** 它进一步延寿到 0.775 s，并使左脚开始卸载；但 26 次切换中最长离地只有 21 ms，尚未形成稳定单支撑。
4. **继续扩大同类 DSMS NLP 的边际收益不成立。** Phase9 是本轮成本最高的单次实验，却因 shooting continuity 恶化在 intervention 前倒地。它只否定当前配置，不否定显式 liftoff 思路。
5. **目前没有合格 dynamics teacher，不得进入 RL。** 任何“7/8”“接近门”或短窗口存活都不能冒充 teacher。
6. **Phase10 已纠正 actuator 合同。** Phase6/8核心信号在正确映射投影下保留；Phase4/5方法结论撤回为candidate级失败。
7. **Phase11 证明“平滑”本身不等于有效liftoff。** 低维事件模式消除了Phase8 chatter，却退化为全程双脚接触；下一动态表示必须直接约束足端/载荷，而不是只扩同一关节模式预算。
8. **Phase12 证明原足位与右单支撑之间存在幅度冲突。** 当前reset的COM距右足中心约0.402m；虽然lower15任务Jacobian满秩，但固定双足完成全载荷转移的线性化解需要最大4.943rad并越限。下一表示必须允许足位/接触时序共同重构，不能只在原地加task-space feedback。
9. **Phase13 证明该冲突贯穿全段。** 137个单支撑intent帧全部存在COM支撑面外差，且是20–40cm量级；结合足距中位0.801m，说明G1/人体足位、X2 COM路径和contact schedule必须联合重定向。source-height标签不再具备X2硬接触真值资格。
10. **Phase14 排除只改标签的廉价修复。** 78.3%帧需要重标，所有可修单支撑都退化为DS，仍有30帧即使DS也存在最大24.8mm支撑面外差。下一生成器必须实质改变q/root/足位。
11. **Phase15 证明髋外展/足宽是重要但非唯一根因。** 由X2 neutral比例直接缩放hip-roll可消除DS支撑矛盾，却仍不能支持任何原SS帧，并引入约10cm重落地修正；它只能作为联合生成器初值。

## 5. 成本—收益判断

本轮已经呈现明显的边际收益递减：

- 几十秒级 DDR/SBTO 可提供结构性诊断；
- reset 投影成本低、收益明确，应该固化；
- 约 90 秒的 reset-aware SBTO 带来 92 ms 额外生存和真实卸载信号；
- 约 20 分钟的长窗口 DSMS 没有带来收益，反而退化到 0.245 s。

因此后续不能靠“更大 population / 更多 IPOPT 迭代 / 更多并行 agent”机械推进。下一步必须改变表示，使优化器直接控制稳定 liftoff、载荷转移以及 root/centroidal 状态，而不是只加预算。

## 6. 后续低资源工作契约

按设备共享要求，后续默认：

- 主任务持续推进，不再并行开启多条实验路线；
- 默认单进程、低线程、0 GPU，启动前检查其他工友负载；
- 一次只运行一个预注册实验，结束后先裁决再决定下一项；
- 不自动进入 RL/PPO；
- 下一项若获授权，只考虑 `reset + 显式载荷转移/足端事件表示 + sampling` 的单路线小实验；Phase9 DSMS 配置停止。

## 7. 推荐决策

当前排序：

1. **保留并固化 Phase6 reset projection。**
2. **把 Phase8 reset-aware SBTO 作为下一轮唯一研究起点。** 目标从“降低综合 loss”改成制造至少 100 ms 的稳定 swing-off，同时限制支撑足位移与 root acceleration；仍须先做低预算证伪。
3. DDR 暂停：当前小预算保持语义但不能修正 contact geometry。
4. 当前 DSMS 长窗口配置停止：除非采用新转录/多重打靶表示或更可靠求解器，不再加迭代。
5. OmniTrack/特权物理生成器在拿到至少一条可执行 seed 前不启动，避免再次卡在 teacher 本身学不会。

当前备份周期计数：本次百度完整归档之后已完成8个实质任务（Phase10合同纠正、Phase11 name-mapped event SBTO、BASE Phase38连续成功suffix目标审计、BASE Phase39单次sequence-guided bridge、动态Phase12 task-space/load物理前审计、动态Phase13全段支撑裕量审计、动态Phase14 contact-label-only审计、动态Phase15 morphology normalization），距离下一次“10任务”百度大包归档还剩2个；Git小提交可随关键纠正即时推送。

## 8. 产物与恢复边界

完整目录：`/home/humanplus/projects/ZHY/dsms_workspace/phase3b` 至 `phase9_reset_contact_dsms`。

Git 保存：脚本、prereg、MD、小 JSON、候选 NPZ（当前总量较小）和本报告。百度网盘保存同一冻结快照压缩包；manifest 必须记录 Git commit、远端路径、字节数和 SHA-256。

## 9. 归档状态

- Git 内容 commit：`6af9e0ac50e2bc48f131698da10111d3d310d084`。
- 百度文件：`HUMAN+/HUMANPLUS_X2_PLUS/2026-08-11/dynamic_retargeting/x2_dynamic_retargeting_race_full_20260811.tar.gz`。
- 字节数：`22,786,040`。
- SHA-256：`219a1dc5739bfdd5078c656055f04df151d225202ee76ef6a2541ef788bde683`。
- 上传状态：`bdpan upload` 已返回成功；远端存在性/大小/下载回读 SHA 按项目约定留给每日人工检查。
