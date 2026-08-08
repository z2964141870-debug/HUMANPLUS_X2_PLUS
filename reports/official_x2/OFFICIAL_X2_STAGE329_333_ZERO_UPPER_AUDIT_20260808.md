# X2 官方 MuJoCo Stage329–333：零上肢课程与条件化适配审计

日期：2026-08-08

基线：Stage306 `s2652`

仿真：AimDK v1.0 官方 X2 MuJoCo
范围：仅训练小型 transition adapter；冻结 SONIC 基座；不上真机。

## 总体判断

本轮没有得到可替换 Stage306 的新 checkpoint。训练中加入真实的 zero-upper 环境后，早期 checkpoint 在小样本中一度达到 `3/3`，但独立复验仍为 fixed `3/5`、fast `4/5`；进一步让 transition adapter 显式读取上肢活动量，三个 checkpoint 也仅为 `1/3、1/3、2/3`。起步和直行在所有有效评估中都通过，失败高度集中在 stiff `1.2x` 域的行走→停稳阶段，说明当前硬缺口是停车后站立/恢复吸引域，而不是上肢条件不可观测或基本行走能力缺失。

## 实验 1：zero-upper 混合课程（Stage329–331）

**假设：** Stage306 的训练环境虽然随机化了 PD，但所有环境都使用运动中的上肢 reference；官方评估中的完全静止上肢属于训练分布外条件。

**干预：** 从 Stage306-s2652 续训小型 transition adapter；64 个环境中 50% 使用严格零上肢目标，50% 保持运动上肢；PD 范围 `0.9–1.2x`；混合 start/cruise/decelerate/stop；基座 actor 冻结。

**对照：** 原 Stage306 在官方 MuJoCo stiff-fixed 为 `3/5`，stiff-fast 为 `5/5`。

**结果：**

- 训练 reward 与 episode length 在 10 个更新内上升，但不能直接代表官方域通过率；
- checkpoint screen：s2654=`3/3`、s2657=`2/3`、s2660=`2/3`、s2662=`0/3`，呈现明显早升后降；
- 对 s2654 做独立五次复验后，fixed=`3/5`、fast=`4/5`，没有超过 Stage306；
- ONNX 与 PyTorch 输出最大差约 `1.12e-7`，可排除导出误差。

**结论：** zero-upper 数据缺口真实存在，但仅混入零上肢课程不足以形成可靠改进；早期 `3/3` 是小样本假甜点，继续训练还会快速破坏闭环。

**下一步：** 检查同一 transition head 是否因无法区分 fixed/moving upper 而承受冲突梯度。

## 实验 2：upper-activity 条件化 transition adapter（Stage332–333）

**假设：** fixed 与 moving upper 共享同一组 locomotion+gait-phase 输入，adapter 看不到当前上肢活动状态，因此两种动力学条件在同一隐变量上产生冲突更新。

**干预：** 给 transition head 只增加两个部署可得的标量：当前上肢意图幅度与未来上肢变化幅度；不改原始 SONIC observation，不开放基座 actor，不允许全身自由 residual。旧 6-D transition 权重通过插入零列精确迁移到 8-D，保证初始化等价。

**对照：** 同一 Stage306-s2652、相同 zero-upper 50% 课程、相同 PD 域与官方 stiff-fixed 门禁。

**结果：**

- 完整项目测试 57 项通过，16-env smoke 中严格得到 8/16 zero-upper；
- Stage332 导出的 s2653/s2655/s2657 与 PyTorch 最大误差约 `1.12e-7`；
- Stage333 官方 stiff-fixed 三次筛选：s2653=`1/3`、s2655=`1/3`、s2657=`2/3`；
- 每次失败均是 startup=true、move=true、stop=false；倒地时 root z 约 `0.139–0.141 m`、tilt 约 `1.53–1.56 rad`；
- 没有 checkpoint 达到预设 `3/3` 晋级门，因此没有继续做 fixed/fast 各五次扩展。

**结论：** 上肢活动不可观测不是主因。条件化带来最多 `2/3` 的弱信号，但未超过现有基线，也未消除同一权重“停稳/倒地”的双稳态。继续扩大该 adapter 或继续扫 checkpoint 的信息增益很低。

## 当前保留与淘汰

- **继续保留：** Stage306-s2652，nominal 直行 `6/6`、nominal turn `7/8`、stiff-fast `5/5`，仍是当前最可信部署基线。
- **不晋级：** Stage329 所有 checkpoint；Stage332 s2653/s2655/s2657。
- **代码保留但默认关闭：** zero-upper curriculum 与 upper-activity conditioning，均为 opt-in，旧行为保持不变，可用于后续严格对照。
- **权重归档策略：** 本轮无新最佳权重，不上传或保留全部中间 checkpoint；只保留可复现实验所需的训练配置、筛选 JSON 和报告。

## 路线裁决

本轮排除了两个简单解释：

1. stiff-fixed 失败不只是“训练没见过完全静止上肢”；
2. 失败也不只是“adapter 不知道上肢是否在动”。

当前证据更支持：行走策略能完成起步和巡航，但在 stiff 域进入停止状态后缺少足够宽的站立/恢复吸引域。下一条高价值路线应是独立的 stop-to-stand/recovery 后端：从真实 locomotion 停步末态分布初始化，专门训练恢复到稳定站姿，再通过状态机和滞回门切换；而不是继续给同一个 transition adapter 叠加条件或扩大 LoRA。

## 下一阶段最小可证伪实验

**假设：** 若失败来自站立吸引域不足，那么专门从 locomotion stop-state 初始化的 stand/recovery policy，应能在不改变行走 actor 的情况下显著提高 stiff-fixed 停车成功率。

**干预：** 冻结 Stage306 行走 actor；收集成功与失败边界附近的 stop states；只训练 stand/recovery 后端，并限制切换时的动作跳变。

**对照：** Stage306 单策略 stiff-fixed `3/5`；现有 scratch stand backend 的同态切换表现；新 recovery backend。

**晋级门：** 独立重复至少 stiff-fixed `5/5`、stiff-fast 不低于 `5/5`，nominal fixed/fast 均不回退，且切换 action delta/jerk 受控。

在该门通过前，不解锁 Any2Any 级长训，也不宣称 X2 后端完成。
