# 长期 Agent 任务卡：DC-PEFT 仿真研究

版本：v1.0  
日期：2026-07-26  
任务性质：长期、纯仿真、单变量推进、证据优先  
工作名称：

> **DC-PEFT：Objective-Decomposed Critics for Cross-Embodiment Whole-Body Policy Adaptation**

中文暂名：

> **面向跨具身全身策略参数高效迁移的目标分解 Critic**

---

## 0. 一句话任务

在不使用真机的条件下，验证：

> 把 locomotion、upper-body imitation 和 whole-body/style regularization
> 分给不同 critic，能否让 G1→X2 的 LoRA/PEFT 迁移既学习 X2 的平衡和接触动力学，
> 又减少对原有上肢跟踪能力的破坏。

这不是“复刻 CWI”。CWI 已经提出多 critic；本项目要研究的是：

```text
CWI 的 objective-decomposed critic
               +
Any2Any/SONIC 式跨具身 PEFT
               +
源策略能力保持与四域物理门禁
```

如果只实现三个 critic，却没有跨具身迁移、能力保持和严格因果对照，不构成新的科研贡献。

---

## 1. 为什么值得做

此前 X2 迁移训练实际始终使用单 critic：

- Stage152-B 配置为 `num_critics: 1`；
- locomotion、root/foot、upper-body、whole-body imitation 和正则项进入同一个总 reward；
- 框架虽然预留多维 value、分维 GAE 和 advantage normalization，但没有真正给 X2 训练提供分组 reward。

因此可能存在：

```text
速度/平衡变好
      ↘
       单一 value target → advantage 混叠 → actor 更新方向不稳定
      ↗
手腕/上肢跟踪变差
```

CWI 把 locomotion、upper-body 和 style 分别交给不同 value function，并使用分组
GAE 与归一化。[CWI 论文](https://arxiv.org/abs/2606.27676)

2026 年另一项 G1 loco-manipulation 对照研究也报告，dual critic 相比 unified critic
在其任务中具有更高的到达效率和吞吐量，并强调普通训练 reward 可能看不出差异，必须使用标准化评价。
[Critic Architecture Matters](https://arxiv.org/abs/2606.11891)

这说明该方向值得做，但不能提前假定它一定能解决 X2：

- 它可以改善 critic 的目标混叠；
- 它不能自动修复动力学不可执行的 reference；
- 它不能凭空生成正确的 COM 转移和足底接触；
- 最终仍是同一个 actor，各目标梯度仍可能冲突。

---

## 2. 科研方向地图与优先级

### S 级主线：跨具身 PEFT 的 decomposed critic

研究 single actor 在 G1→X2 LoRA 微调时，分目标 critic 是否能降低价值混叠和
source capability forgetting。这是本卡唯一默认允许进入训练的方向。

### A 级竞争解释：数据使用解耦与 reference feasibility

CWI 不只拆 critic，也让多样上肢数据和精选稳定下肢数据承担不同角色。若问题来自
下肢 reference 本身不可执行，critic 分解不会治本。因此必须保留两种竞争解释：

1. upper/lower 数据混合方式不合理；
2. X2 reference 缺少动力学可行性。

第一轮保持数据不变以隔离 critic 因果；只有 multi-critic 被证伪或 ideal 域仍失败，
才允许转向数据使用解耦或 reference repair。

### B 级证据解锁：actor 梯度冲突处理

如果多 critic 提高 value quality，但目标 actor-gradient 仍持续负余弦，再测试
[GCR-PPO](https://arxiv.org/abs/2509.14816) 或 PCGrad/CAGrad 类方法。
没有梯度冲突证据时禁止添加。

### B 级证据解锁：约束式目标聚合

若 locomotion/safety 仍被 upper/style 自由交换，才将安全目标改为 constraint，
参考 [Constrained Style Learning](https://arxiv.org/abs/2507.09371)。

### C 级未来方向

- contact/phase/COM/DCM privileged critic；
- preference-conditioned Pareto policy；
- masked-goal 多设备策略；
- morphology-conditioned critic/graph policy；
- recovery policy switching。

这些不进入第一轮最小实验。

---

## 3. 工作范围与安全红线

### 允许

- 阅读现有 SONIC/X2 代码、配置、checkpoint 和日志；
- 在仿真中运行训练、回放和评价；
- 在新工作目录实现 overlay、补丁和测试；
- 使用 IsaacLab/MuJoCo 已有模型估计的 contact、COM、DCM 等训练诊断；
- 生成新 checkpoint、JSON、Markdown 和必要视频；
- 清理由本任务新生成且已经被判定淘汰的中间 checkpoint。

### 禁止

- 连接、使能或驱动真实机器人；
- 调用 X2 真机 command topic/service；
- 修改机器人固件、SDK、系统驱动或全局 Python 环境；
- 把仿真 contact force、COM、DCM 称为实机真值；
- 修改或删除旧 X2 工程中的原始 checkpoint、数据集和历史报告；
- 为了得到正结果后验改变门禁、删除失败 seed 或只汇报最好 checkpoint；
- 未通过阶段门就直接启动 3000/8000 iteration 长训；
- 一开始同时加入 multi-critic、PCGrad、privileged critic、phase input 和新 reward。

### 写入位置

所有新代码、报告和实验索引优先写入：

```text
/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
```

旧工程：

```text
/home/humanplus/x2_teleop_final/x2_sonic
```

默认只读。若无法通过 overlay 完成，先在新目录保存精确补丁和源文件 SHA256；
不得直接覆盖旧文件。

---

## 4. 绑定基线

### 主要 checkpoint

```text
/home/humanplus/x2_teleop_final/x2_sonic/logs/ppo_dryrun/
x2_stage152_B_pilot_seed0_v1/model_step_000200.pt
```

该 checkpoint 不是被预先宣布为“最终最好模型”，而是本研究的冻结比较起点。
Agent 必须在 Phase 0 重新计算 SHA256，并使用统一评价确认其实际状态。

### 代码事实基线

- Stage152-B：`num_critics: 1`；
- PPO storage 已支持 `reward/value/return/advantage` 的 `num_critics` 维度；
- PPO 已支持逐 critic advantage normalization；
- `multi_critic_advantage_weights` 分支存在，但旧实验未启用；
- 当前仍需验证环境是否真正输出分组 reward，而不是把同一个 scalar 广播三次。

### 初始数据

第一轮禁止继续增加数据集。使用相同数据做严格 A/B：

- 现有 X2 reference；
- 已有官方/遥控步态与组合动作；
- Stage152 既有训练 motion 集；
- 至少包含：
  - 静态上肢；
  - 原地上肢＋重心扰动；
  - 行走；
  - 转向；
  - 下蹲；
  - 行走＋上肢组合动作；
  - held-out 动作。

---

## 5. 核心假设

### H1：价值估计干扰

单 critic 的 TD residual 和 value error 被不同目标的尺度、方差和时序混合；
分 critic 后，各目标 explained variance 提高，TD residual 方差下降。

### H2：能力保持

相同训练预算下，decomposed critic 能让 X2 locomotion 改善，同时减少
upper-body tracking 退化。

### H3：语义分组

收益来自正确的目标分组，而不仅是增加 value 输出头或改变梯度尺度。

### H4：有限作用

若 reference 在 X2 上动力学不可执行，multi-critic 只能改善优化稳定性，
不会自动解决完整卸载—抬脚—落脚闭环。

---

## 6. Reward Contract

Agent 不得凭名字随意拆 reward。必须先导出所有实际激活 reward term、权重、均值、
方差和作用身体部位，再确定唯一归属。

### 最小 dual-critic 分组

#### `V_loco`

负责：

- survival/termination；
- root position/orientation；
- velocity/heading/height；
- foot contact、swing clearance、stance slip；
- COM/DCM、支撑与载荷转移；
- 腿、腰的安全和动力学正则。

#### `V_upper`

负责：

- wrist/hand target position；
- elbow/shoulder/upper-body joint tracking；
- torso-relative upper-body tracking；
- 上肢平滑、关节限制和必要正则。

### 可选 triple-critic 第三组

只有 dual critic 通过阶段门后，才允许增加：

#### `V_imitation_style`

负责：

- whole-body relative pose/orientation imitation；
- 与任务无直接关系的动作自然度；
- reference velocity/style consistency；
- action smoothness 中真正属于全身风格的部分。

当前没有可靠 AMP discriminator 时，必须称为
`imitation_style_regularization`，不得声称复现了 CWI 的 adversarial style critic。

### Reward Contract 硬门

每个 active reward term 必须：

1. 恰好属于一个 group；
2. 保存归属理由；
3. 保留原始 term weight；
4. 在同一 fixed rollout 上满足：

```text
r_scalar == r_loco + r_upper (+ r_imitation_style)
```

逐环境、逐时间步最大绝对误差必须 `≤ 1e-6`。  
不通过时禁止训练。

---

## 7. 对照矩阵

### 必做

| 编号 | Actor | Critic | Reward | 用途 |
|---|---|---|---|---|
| B0 | 冻结 Stage152-B | 原 critic | 原 scalar | 零训练能力基线 |
| B1 | 同一 LoRA/PEFT 范围 | single critic | 原 scalar | 训练控制组 |
| E1 | 与 B1 完全相同 | dual critic | loco + upper | 主实验 |
| N1 | 与 E1 完全相同 | dual critic | 错误/随机语义分组 | 负控制 |

### 证据解锁后再做

| 编号 | 条件 | 实验 |
|---|---|---|
| E2 | E1 明确优于 B1/N1 | triple critic |
| E3 | 测得持续负梯度余弦 | E1 + GCR-PPO/PCGrad 类融合 |
| E4 | locomotion value 仍不可辨识接触阶段 | phase/contact privileged critic |
| E5 | 稳定性仍被其他目标牺牲 | constrained/lexicographic advantage |
| E6 | ideal 域仍无法执行同一 reference | reference repair/data-use decoupling |

禁止把 E2–E6 同时启用。

---

## 8. 必须保持相同的变量

B1、E1、N1 必须保持：

- 同一 actor 初始化；
- 同一 LoRA 注入范围；
- 同一训练 motion 列表；
- 同一 reference sampling；
- 同一 actuator domain；
- 同一 action scale、PD、仿真步长和 decimation；
- 同一 PPO iteration、rollout、batch 和 epoch 数；
- 同一 actor/critic learning rate；
- 同一 KL/anchor/source-retention 约束；
- 同一 termination；
- 同一 reward term 与总权重；
- 同一随机种子集合；
- 同一评价脚本和门禁。

允许变化的主变量只有 critic/reward decomposition。

Multi-critic 将多个标准化 advantage 相加时会增大 actor 梯度尺度。初始建议：

```text
dual critic:   [0.5, 0.5]
triple critic: [1/3, 1/3, 1/3]
```

必须记录实际 actor gradient norm；若与 B1 差异过大，先做尺度匹配，
不得把更大的有效学习率误认为结构收益。

---

## 9. 分阶段任务

每个阶段结束必须按以下格式写入 `STATUS.md`：

```text
假设：
干预：
对照：
结果：
结论：
下一步：
```

### Phase 0：只读审计与基线冻结

- [ ] 记录 GPU、磁盘、环境和项目大小；
- [ ] 计算 checkpoint、配置、trainer、评价脚本 SHA256；
- [ ] 确认旧实验确实 `num_critics=1`；
- [ ] 复跑 B0，不训练；
- [ ] 固定动作清单、四域和评价门；
- [ ] 输出 `baseline_manifest.yaml` 与 `baseline_report.md`。

门禁：

- checkpoint 可重复加载；
- 两次 deterministic evaluation 指标误差在预设容差内；
- 不得因基线难看而更换评价动作。

### Phase 1：Reward Contract 与固定 rollout

- [ ] 导出 active reward term 清单；
- [ ] 设计 dual 分组；
- [ ] 实现 vector reward；
- [ ] 证明 vector sum 与 scalar reward 逐步等价；
- [ ] 记录各组 reward 均值、标准差、相关系数和数量级；
- [ ] 给错误分组 N1 预注册固定 seed。

门禁：

- scalar/vector 最大绝对误差 `≤1e-6`；
- 无 term 遗漏、重复或动态改变归属；
- fixed rollout 完整可回放。

### Phase 2：Multi-critic 基础实现

- [ ] value 输出从 1 维扩展为 2 维；
- [ ] 每头单独 return、GAE 和 normalization；
- [ ] actor 使用固定权重合成 advantage；
- [ ] 保存每头 value loss、explained variance、TD residual；
- [ ] checkpoint 保存/恢复兼容；
- [ ] `num_critics=1` 时行为与旧路径一致。

门禁：

- 单 critic compatibility test 全通过；
- actor 在更新前与 B0 输出逐位一致；
- reward/value/advantage shape 无 silent broadcast；
- 训练一步后所有梯度有限，无 NaN/Inf。

### Phase 3：一阶与短训因果验证

依次做：

- 1 iteration；
- 5 iterations；
- 25 iterations。

同时运行 B1、E1、N1，记录：

- actor/critic gradient norm；
- 各组 advantage mean/std；
- 各组 actor gradient cosine；
- KL；
- action delta；
- source policy retention；
- value explained variance；
- fixed replay 上的 action drift。

停止条件：

- E1 的有效 actor 学习率明显大于 B1 且无法尺度匹配；
- 发现 reward 广播、head collapse 或 checkpoint 错载；
- upper 或 loco 任一目标立即灾难性退化。

### Phase 4：200-iteration Pilot

通过 Phase 3 后，运行：

```text
B1 single critic × seeds {0, 1, 2}
E1 dual critic   × seeds {0, 1, 2}
N1 bad grouping  × seeds {0, 1, 2}
```

禁止只比较一个 seed。

每 25/50 iteration 保存评价摘要；checkpoint 只保留：

- step 0；
- 25；
- 50；
- 100；
- 200；
- 当前 best；
- 首个明确退化点。

### Phase 5：1000-iteration 中训

只有同时满足以下条件才解锁：

1. E1 在 3 个 seed 中至少 2 个方向一致；
2. 相比 B1，E1 不是单纯牺牲 upper 换 loco，或牺牲 loco 换 upper；
3. N1 不应与 E1 同样改善；
4. value/advantage 诊断支持 critic 干扰下降；
5. 没有隐含 reward、learning-rate 或参数量混淆。

中训只保留 B1 和 E1，不再继续明显失败的 N1。

### Phase 6：3000-iteration 长训

1000 iteration 后若曲线仍震荡上升、没有出现能力保持恶化，允许运行 3000。

不得只看 mean reward。长训必须同时观察：

- locomotion 门；
- upper tracking；
- imitation/style；
- episode length；
- termination phase；
- 四域泛化；
- source retention；
- checkpoint 曲线。

若 1000 iteration 已无增长或持续退化，不因“Any2Any 训了更久”而机械续训。

### Phase 7：证据解锁扩展

按诊断只选择一个：

- 目标梯度持续负余弦 → gradient surgery；
- 稳定性目标仍被自由牺牲 → constrained/lexicographic；
- locomotion critic 无法识别接触时序 → privileged phase/contact critic；
- dual 有效且第三组定义可信 → triple critic；
- ideal 域 reference 仍不可执行 → data-use decoupling/reference repair。

新增方法必须重新设置 matched control。

### Phase 8：最终评估与科研判断

- [ ] 汇总 3 seeds；
- [ ] 四域测试；
- [ ] held-out 动作；
- [ ] 失败 phase 分类；
- [ ] Pareto 对比；
- [ ] 生成可肉眼检查的 reference/B0/B1/E1 视频；
- [ ] 输出正结果或负结果；
- [ ] 明确是否值得继续形成论文。

---

## 10. 四个独立评估域

必须分别报告，禁止混合：

1. `ideal actuator`
2. `nominal actuator`
3. `nominal + delay`
4. `nominal + delay + noise`

如果 ideal 域都不能稳定，优先判定训练/目标问题；  
如果 ideal 改善而 nominal 失败，优先判定执行器鲁棒性问题；  
如果 nominal 通过、delay 失败，才讨论时序补偿。

---

## 11. 评价指标

### Locomotion

- episode survival / mean length；
- root position/orientation error；
- 世界前进方向、横漂、航向；
- velocity/height tracking；
- 完整 DS→SS→DS 接触周期；
- swing-foot liftoff 与 clearance；
- stance-foot slip；
- termination phase；
- 仿真 COM/DCM 与支撑关系；
- action saturation、torque、joint limit。

### Upper body

- 左右 wrist position/orientation error；
- elbow/shoulder tracking；
- torso-relative tracking；
- combined locomotion+upper success；
- 动作延迟；
- source-policy action retention。

### Imitation/style

- relative body pose/orientation；
- joint/body velocity consistency；
- action smoothness；
- 若无 AMP，禁止汇报“discriminator style score”。

### Critic/optimization mechanism

- 每头 value explained variance；
- 每头 TD residual mean/std；
- 每头 value loss；
- advantage mean/std/tail；
- 分组 advantage correlation；
- 分组 actor-gradient cosine；
- actor gradient norm；
- KL 与 LoRA weight drift；
- head collapse 比例。

### 综合判断

同时报告：

- 每项目标相对 B0/B1 的变化；
- worst-objective regression；
- Pareto dominance；
- 失败率和最坏 seed；
- 不仅报告加权总 reward。

---

## 12. 成功门

### 最低工程成功

- 完成可靠 vector reward；
- single critic compatibility 保持；
- multi-critic checkpoint 可训练、保存、恢复和评价；
- 无 silent broadcast 或数据口径错误。

### 方法有效

E1 相比 B1 必须满足以下任一条件，并在至少 2/3 seeds 一致：

1. locomotion 严格通过率提高，同时 upper error 恶化不超过 5%；或
2. upper tracking error 降低至少 10%，同时 locomotion 严格通过率不下降；或
3. 两者均有改善，并在四域 worst-case 指标上形成 Pareto dominance。

同时：

- E1 必须优于错误分组 N1；
- 改善不能仅来自更大 actor gradient；
- value/advantage 诊断至少支持一个预注册机制假设。

### 有科研价值

至少需要：

- matched single/dual critic 对照；
- semantic/bad grouping 负控制；
- 3 seeds；
- 四域与 held-out；
- 能力保持指标；
- 失败案例；
- 明确说明 multi-critic 不能解决的 reference 动力学问题。

---

## 13. 失败与停止条件

以下任一情况成立，应停止扩大训练：

- reward-vector 等价性无法成立；
- single critic compatibility 被破坏；
- E1 与 N1 无差别，说明语义分组没有证据；
- E1 只通过牺牲另一个目标改善；
- 3 seeds 方向不一致且置信区间高度重叠；
- value quality 没改善，actor 梯度冲突也没下降；
- 1000 iteration 后无稳定上升；
- 改善只存在于训练动作，不存在于 held-out；
- 所有收益都能由有效学习率或参数量解释；
- 当前 reference 在 ideal 域也物理不可执行。

失败不是任务失败。必须输出：

- 哪个假设被推翻；
- 最小反例；
- 为什么继续训练价值低；
- 哪个相邻方向仍值得测试。

---

## 14. 磁盘与运行规则

- 开始前记录剩余磁盘和项目大小；
- 同一时刻最多一个 GPU 训练进程；
- 所有进程必须设置最大 iteration；
- 禁止无限日志和默认视频录制；
- 每次评价只渲染必要的代表动作；
- 中间 checkpoint 在指标落盘并记录 SHA256 后方可清理；
- 只允许清理由本任务在新目录生成的文件；
- 永久保留 baseline、best、final、首个明确失败、配置、指标、manifest、
  关键视频和最小反例。

---

## 15. 长期 Agent 工作规则

1. 不需要人工参与时持续推进。
2. 每完成一个阶段及时更新文件，不依赖聊天记录。
3. 训练运行期间降低轮询频率，避免无意义消耗。
4. 不把“GPU 在跑”当作进展；进展必须对应可验证假设。
5. 每次只改变一个主要因素。
6. 发现评价器漏洞时，先修复并重评旧结果。
7. 不允许选择性汇报。
8. 不允许因为短训下降立即判死刑；但进入长训必须满足阶段门。
9. 若连续两个阶段没有信息增益，先全局复盘，不继续扫参数。
10. 只有在以下情况停止并汇报：
    - 获得明确改善；
    - 达到当前甜点位；
    - 出现需要用户授权的新范围；
    - 方法已被充分证伪；
    - 环境/GPU/磁盘形成真实阻塞。

---

## 16. 必须维护的文件

```text
CWI_CrossEmbodiment_Sim/
├── TASK_CARD.md
├── STATUS.md
├── EXPERIMENTS.md
├── DECISIONS.md
├── FAILURES.md
├── DISK_LEDGER.md
├── SAFETY_LOG.md
├── manifests/
├── patches/
├── src/
├── tests/
├── results/
├── reports/
└── videos/
```

每个实验使用：

```text
假设：
干预：
对照：
结果：
结论：
下一步：
```

---

## 17. 第一项工作

Agent 接到本卡后，不应立即训练。第一项工作是：

> 只读审计 Stage152-B 的实际 reward、critic、checkpoint 和评价入口，
> 生成 reward-group candidate 与 scalar/vector equivalence 测试方案。

Phase 0/1 完成前，不允许启动长期训练。

---

## 18. 最终交付

长期 Agent 最终必须交付：

1. 可复现 multi-critic 实现；
2. single-critic compatibility test；
3. scalar/vector reward ContractMR；
4. B0/B1/E1/N1 三 seed 证据；
5. 200/1000/必要时 3000 iteration 曲线；
6. 四域和 held-out 报告；
7. reference/B0/B1/E1 对比视频；
8. 完整失败案例；
9. 科研结论：
   - 值得写论文；
   - 只值得保留为工程模块；
   - 或该方向在当前 X2 迁移上无效。

---

## 19. 当前任务状态

```text
任务卡：READY
真机：DISABLED
长期训练：LOCKED，等待 Phase 0/1
主实验：single critic vs semantic dual critic
负控制：bad/shuffled reward grouping
可选扩展：证据解锁
```
