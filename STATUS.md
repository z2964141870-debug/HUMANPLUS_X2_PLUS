# DC-PEFT 仿真研究状态

## Phase 0/1（完成）

假设：
Stage152-B 的优化干扰可能部分来自单 critic 混合估计 locomotion 与 imitation/upper 目标；先建立完全等价的 reward vector，才能因果检验。

干预：
只读冻结 Stage152-B；在新工程通过 import overlay 重组 IsaacLab 已计算的逐项 reward，不修改旧 X2/SONIC 工程。

对照：
原 scalar reward、语义 dual grouping、固定 seed 的 shuffled grouping。

结果：
已确认旧配置 `num_critics=1`；旧 trainer 具备多维 storage、逐头 GAE/normalization 和 advantage 加权，但环境原本只输出 scalar。GPU 内核模块为 595.71.05，用户态库为 595.84，NVML 失效；PyTorch 与 Isaac physics 仍能使用 CUDA。静态单测 3/3 通过。真实 X2 Isaac smoke 完成 8 步，reward scalar/vector 最大绝对误差为 `1.4901161e-08`，小于 `1e-6` 硬门；两组平均单步 reward 分别约 0.02023 与 0.07542。Stage152-B 四域基线完成两次独立复跑，逐动作 7 项连续指标最大绝对差为 `0.0`；两轮门禁均为 ideal `0/4`、filter `1/4`、delay `0/4`、noise `0/4`。

结论：
历史工作没有真正检验过 CWI 式多 critic。Reward Contract 的数学与运行时链路及 B0 可重复性均已通过。B0 很弱但稳定，适合作为因果对照；当前第一风险转为双头 critic 如何从单头 checkpoint 公平初始化。

下一步：
生成单头→双头 critic 的显式复制初始化；验证 actor action 完全不变、两个 value head 的形状与载入一致，再解锁 B1/E1/N1 的 1/5/25 步级联。

## Phase 2 / Phase 3-I1（完成）

假设：
将 Stage152-B 的 scalar value 等分为两个初始 head，可在不改变 actor 和总 value 的前提下，为 semantic/shuffled decomposition 提供公平 warm start。

干预：
只扩展 critic 最后一层；policy 90 个 checkpoint tensor 原样复制。E1 使用语义分组，N1 使用 seed `20260728` 预注册的等尺寸随机分组；两者 advantage 均为 `[0.5,0.5]`。

对照：
B1 single scalar；E1/N1 使用相同双头初始化。三组均为 seed 0、32 env、24 steps、3 PPO epochs、4 minibatches、相同 Stage152 actuator mixture、相同 actor/critic LR。

结果：
静态与奖励单测 6/6 通过。双头初始化与 B0 的 1040-step rollout 在 402,480 个数值字段上完全一致，其中 action 数值 68,640 个；value checkpoint 45/45 无错位。B1/E1/N1 一步训练均无 NaN/Inf，actor preclip gradient 分别为 `7.753/7.648/7.623`，KL 为 `0.001533/0.001511/0.001527`。E1/N1 reward reconstruction 最大误差为 `1.49e-08/2.24e-08`。E1 保存后的双头 checkpoint 已通过 8-step 重载，45/45 tensor 匹配。

结论：
Phase 2 基础实现与一步数值门通过；没有 reward broadcast、head shape 错位或双 critic 有效学习率放大的证据。一步结果只证明机制健康，不证明 E1 性能更好。

下一步：
从相同初始 checkpoint 独立运行 B1/E1/N1 的 5 iterations；先比较曲线、actor drift、逐 head EV/TD 与梯度尺度，再决定是否解锁 25 iterations。

## Phase 3-I5（完成）

假设：
若目标分解主要改善早期优化稳定性，E1 应在相同五轮预算下少于 B1 的能力破坏，并开始与随机分组 N1 分化。

干预：
B1/E1/N1 各自从同一 Stage152-B 起点独立训练 5 iterations；训练后使用完全相同的四域 4-motion 门禁。

对照：
所有物理、采样、PPO、LoRA 和 evaluation 设置相同；E1/N1 共享相同双头初始化和 `[0.5,0.5]` 权重，仅 reward term 归属不同。

结果：
训练第 5 轮 episode length 为 B1 `28.07`、E1 `30.06`、N1 `28.93`。四域 strict 总通过数为 `0/16、1/16、2/16`。E1 的 delay 域连续指标最好（root XY RMSE `0.184 m`、progress-ratio error `1.827`，4/4 stable/contact），但 ideal 过冲仍差；N1 ideal 最好且过 1 条，但 delay/noise 各出现 1 条跌倒，root RMSE 为 `0.417/0.345 m`。三组训练均有限，reward reconstruction 仍小于 `1.2e-7`。

结论：
五轮结果支持“multi-critic 比 single 更少立即破坏”的弱信号，但不支持“语义分组已经优于随机分组”。N1 的 ideal 优势与 E1 的 delay 鲁棒性形成明显 trade-off，离散通过数不足以裁决。

下一步：
解锁 25-iteration matched gate。只有 E1 在更长预算中相对 B1 和 N1 形成一致的 locomotion/upper/robustness 改善，才进入 200-iteration 多 seed；否则将该方向判为负结果或仅优化稳定性工具。

## Phase 3-I25（完成，达到当前甜点位）

假设：
语义 dual critic 若真正缓解目标干扰，应同时改善 value quality、四域物理门禁和 upper-body preservation，并优于 shuffled grouping。

干预：
B1/E1/N1 均从 Stage152-B 独立训练 25 iterations，随后运行完全相同的 ideal/filter/delay/noise 4-motion panel。

对照：
B1 single scalar、E1 semantic dual、N1 fixed-seed shuffled dual；初始化、actor、PPO、LoRA、动作集和计算预算匹配。

结果：
E1 的 episode reward/length 为 `5.381/76.42`，高于 B1 的 `4.950/71.55` 和 N1 的 `4.343/62.40`；两个语义 value head 的 EV 为 `0.832/0.830`，明显好于 N1 的 `0.734/0.647`。但四域 strict 为 B0/B1/E1/N1=`1/0/1/0`（各 16 条），E1 stable 仅 `12/16`，低于 B0/N1 的 `16/16`。E1 wrist mean 为 `71.07 mm`，相比 B0 `56.23 mm` 恶化 `26.38%`，超过 5% 能力保持门。

结论：
工程链路和“语义分解改善 critic 估计”得到证实，但“改善 critic 即改善 X2 物理迁移”被否定。E1 没有 Pareto dominance，并以明显上肢退化换取局部训练统计改善。按预注册停止条件，不解锁 200×3 seeds。

下一步：
冻结当前结果，先做 objective-to-actor gradient cosine 的无训练/最小训练诊断。只有观测到持续负梯度冲突，才证据解锁 PCGrad/GCR-PPO；否则停止 CWI critic 主线，返回 reference feasibility 与 data-use decoupling。

## 第二轮：Actor Gradient Conflict（完成）

假设：
E1 的 value quality 变好但上肢退化，可能源于 locomotion 与 imitation/upper advantage 在共享 actor 上产生持续负梯度冲突。

干预：
新增不修改旧工程的逐 head actor-gradient probe；完成 E1/N1 × INIT/I25 × 3 seeds 的 12 格零更新矩阵，并完整复跑 E1 25 轮记录 300 个 microbatch。

对照：
语义分组与 shuffled 分组、初始 checkpoint 与 I25 checkpoint、三个环境 seed；边界 probe 的 optimizer 显式归零并检查 actor 参数逐位不变。

结果：
12 个边界 probe 的 48/48 microbatch 全为正，E1-I25 最低 cosine `0.771`。完整 E1 轨迹 aggregate cosine 均值 `0.897`、最低 `0.647`；300 个 microbatch 仅 2 个为负（0.67%），25/25 轮的 aggregate gradient 全为正。探针最大 policy-loss 重构误差 `8.94e-08`，边界 actor 漂移严格 `0.0`，带探针复跑与原训练核心指标逐轮一致。

结论：
“持续 actor 梯度冲突导致 E1 失败”被推翻。PCGrad/GCR-PPO 在当前轨迹中几乎无操作空间，不解锁。当前更可能的问题是 objective 不纯、upper 数据覆盖不足，以及训练 reward 不能充当 held-out upper capability constraint。

下一步：
若继续 CWI，转做 data-use decoupling / pure objective contract 的最小因果实验；否则返回 reference feasibility。禁止继续扩大 critic 或盲目换 actor 优化器。

## 第三轮：Data-use Decoupling + Action Ownership（完成）

假设：
此前 gait 中的大幅手腕退化可能是腿/根闭环失稳的连带现象；若限制新 LoRA 只能修改腰腿动作，或许能保护上肢并使 nominal/delay 域持续改善。

干预：
先用 3 条 fixed-feet 上肢动作隔离能力，再将 Stage152-B LoRA 物化为共同基座；从完全相同起点训练 FBP（29 个非头 DOF）与 LBP（15 个腰腿 DOF），在 decoder 边界投影整个适配 action delta。完成 5-step 与 25-step 四域对照。

对照：
FBP/LBP 的数据、critic、学习率、PPO、执行器混合与初始化完全匹配；首步 4×31 mean action 数值差严格为 0。

结果：
fixed-feet filter 中 E1 相比 B0 的 wrist mean 仅 `+1.78%`，证明原大幅腕误差主要是 locomotion 连带效应。LBP-5 在 delay 域将 root RMSE `0.240→0.176 m`、progress error `2.870→1.206`、contact `2/4→4/4`，且四域上肢均保持在 5% 内；但 LBP-25 的 delay stable 退到 `2/4`、root RMSE 升至 `0.402 m`，总 strict 为 `0/16`。训练 reward/length 上升，而 reward-per-step 仅在约 `0.063–0.071` 横盘。

结论：
动作所有权是有效安全 contract，并在短窗出现 nominal+delay 正信号；但它不能单独修复长训目标错配。当前 reward 的上升主要来自 survival，而非 held-out 世界步态变准。禁止解锁 200+ steps。

下一步：
保留 LBP 投影；训练选择改为固定起点四域早停，并设计不把高频/不可信 raw root 当硬真值的带宽受限路径或 contact/task-command 目标。

## 第四轮：X2 原生下层 × 上肢意图接口（完成，得到新甜点位）

假设：
放弃全身 strict reference，让 X2 原生策略独占腿和腰、上层只控制肩肘腕，
可以绕开此前 root/foot/upper 相互竞争；若延迟域仍失败，应先修控制契约时序。

干预：
冻结正式 Stage208-s2550 下层，注入 fixed-feet wave 的 14DOF 上肢 oracle；
完成零幅等价、0.25/0.50 幅值、ideal/filter/delay、手臂/腰部零延迟反事实，
以及 0.5× 上肢时间轴整形。全程未训练、未修改旧工程、未生成 checkpoint。

对照：
同一 seed、0.30 m/s、10 秒、碰撞/执行器/下层完全匹配的静态上身 control。
注入器 `scale=0` 与旧 evaluator 所有 trace 数组最大差严格为 `0.0`。

结果：
0.25× 原速在 ideal/filter 通过，在 delay 中因侧漂增加 `0.303 m` 失败；
去掉腰部 1 帧延迟后相对退化消失。保持完整 delay、仅将上肢时间轴放慢一半，
0.25× 工作点走满 10 秒，heading `0.510→0.491 rad`、lateral
`0.612→0.505 m`、vx RMSE 改善 9.2%，双脚抬升约 39 mm。0.50× 半速仍因
heading/lateral 退化失败。

结论：
分层架构的物理可行性得到首个正证据，真正需要的 Adapter 至少包含幅值边界
与时序整形；但当前只验证 1 motion/1 seed/1 speed，且绝对侧漂仍高，不能称为
完整 SONIC→X2 或部署模型。

下一步：
不接裸 SONIC、不启长训。先实现有界的上肢安全 Adapter，并扩大 motion/seed/
速度 envelope；通过后才把 oracle 换成冻结 SONIC 上肢输出。

## 第五轮：有界上肢 Adapter 覆盖扩展（完成，达到裁决甜点位）

假设：
第四轮 `scale=0.25, time_scale=0.5` 的正例可通过幅值、速度和健康回退扩成
多动作、多速度的安全接口。

干预：
冻结 Stage208；上肢加入 `0.12 rad` 偏移、`0.20 rad/s` 速度、tilt/height
回退，并测试 4 类动作 × 2 个有效速度。追加仅依赖 torso IMU 的
`0.25 rad` 航向锁存回退。

对照：
每个速度的静态上身 control。seed42/7 在关闭随机化后逐项相同，正式统计不
重复计数。

结果：
正常 `0.30 m/s` 为 3/4 严格通过、4/4 生存；低速 `0.20 m/s` 为 0/4
通过、2/4 生存。真实 wave 在正常速度使 heading/lateral 相对恶化
`0.201 rad/0.223 m`。IMU 回退后低速生存反而由 2/4 降到 0/4，原本生存的
knocking/box_lift 也跌倒。

结论：
分层接口有有限正常步速工作区，但固定 filter 和事后 supervisor 都不能形成
通用安全保证；不解锁真实 SONIC 注入。根因收敛到下层只看到当前 arm q/dq、
看不到未来上肢意图，在长 arm delay 下只能反应式补偿。

下一步：
停止扫幅值/回退阈值；只允许预注册 future-upper-intent + gait-phase 的小型、
低维腰髋协调 Adapter，并以 reactive-only / no-future / no-phase 作因果对照。

## 第六轮：Future-Intent 腰髋协调 Adapter（完成，部分正信号但不晋级）

假设：
冻结 Stage208 后，提前 `0.6 s` 的有界上肢意图与 gait phase 能让小型腰髋
Adapter 预先协调重心，并优于 CURRENT 与 FUTURE-NOPHASE。

干预：
三支同容量 Adapter 从 Stage208-s2550 独立训练 1/5/25 updates；只允许修正
左右髋与 waist yaw/roll，residual 上限 0.10。正式评估为 6 motions ×
0.20/0.30 m/s 的 nominal+delay 400-step 面板。

对照：
BASE、CURRENT、FUTURE、FUTURE-NOPHASE；其中真实 wave 与两个动作不参与
训练。零初始化与旧 Isaac trace 的 39 个数组最大差为 0。

结果：
Gate5 的 FUTURE 与 BASE 都是 5/12 生存，但平均航向/横漂为
`0.614/0.473`，优于 CURRENT `0.643/0.520` 和 NOPHASE `0.657/0.511`。
Gate25 后 FUTURE 总步数最高（3218），平均航向/横漂降到
`0.560 rad/0.282 m`；然而低速总步数 `962→945`，留出
`wave_left@0.30` 航向 `0.510→1.068 rad`、倾角 `0.299→0.509 rad`。

结论：
future+phase 有真实的动作子集方向信号，但没有形成跨动作、跨速度的通用安全
接口。Stage208 保留为基线，FUTURE-s2575 仅作研究候选；禁止用同一目标继续
200+ updates，也不解锁 SONIC 或真机。

下一步：
若开第七轮，必须引入显式方向风险约束、动作模态化或可拒绝介入的 shadow
supervisor，并启用真正随机的多 seed；不能继续扫本轮网络容量和训练轮数。

## 第七轮：保守 Adapter + 风险约束 + 多种子扰动（完成）

假设：
缩小 Future-Intent residual、加入世界 heading 风险项或进行保守短训，可能
保留 Stage6 的方向收益并消除留出回归。

干预：
先对冻结 FUTURE-s2575 扫 residual blend 0.25/0.50/0.75；再从 Stage208
独立训练 NEUTRAL05 与 heading-risk RISK05 各 5 updates。对意外出现的
NEUTRAL05 候选使用 seed17/42/73、随机初始姿态/速度的 36-case 成对面板，
并追加逐步轨迹的共同生存窗口诊断。

对照：
Stage208 BASE；所有成对初态 quaternion、倾角、线/角速度最大差 0.0。
Stage7D 逐步记录版与 Stage7C 汇总指标最大差也为 0.0。

结果：
RISK05 未优于 NEUTRAL05，停止。NEUTRAL05 在 36-case 中将生存
`15→17`、总步数 `9674→9961`、低速步数 `2788→2841`，世界 heading
`0.5988→0.5697 rad`、tilt `0.5465→0.5111 rad`，但 lateral
`0.4657→0.4872 m`。共同窗口 lateral max 略好 `0.458858→0.456815 m`，
而 lateral 时间均值仍略差 `0.178080→0.180017 m`；一个共同生存 wave
案例横漂回归 0.1513 m。

结论：
Future+phase 的局部闭环收益跨随机初态复现，说明迁移仍有研究希望；但当前
全程 residual 的收益在动作/初态间交换，未形成严格鲁棒 Pareto。Stage208
继续作为正式基线，NEUTRAL05 只作互补研究候选，当前 objective 停止。

下一步：
只有独立预注册的 predictive gate / mode-conditioned selective intervention
值得继续：先预测 BASE 与候选谁更安全，再选择性介入；禁止继续同目标长训。
