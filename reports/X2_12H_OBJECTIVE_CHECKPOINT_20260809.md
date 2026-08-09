# X2 12 小时目标推进检查点

日期：2026-08-09
范围：官方 AimDK v1.0 MuJoCo、速度型 BASE、faithful Any2Any、分层上肢遥操作
裁决：**达到证据甜点，暂停继续扩训**

## 一句话结论

本轮没有得到可晋级的全身遥操作模型，但解决了“官方闭源仿真与 direct MuJoCo 自相矛盾”的核心方法学问题，并首次证明“closed AimDK 原生腿腰后端 + 有界人体上肢”可以完整起步、行走和停车；剩余瓶颈是下肢后端对上肢扰动和真实单支撑步态的鲁棒性，而不是接口或 LoRA 是否能运行。

## 1. 官方域矛盾已闭环

- Phase28 的首帧 observation 与 action 虽一致，direct MuJoCo 仍在首个 20 ms 发生物理速度分叉。
- MuJoCo 3.3.7 与 3.4.0 的冻结 0.2 s 对照逐元素一致，版本不是主因。
- LD_PRELOAD observer 在不修改官方 binary/scene/controller 的前提下，无扰动记录 closed AimDK 内部 `mjData`；hook on/off 的 qpos/qvel 逐步 bitwise 一致。
- 首个 93D stand telemetry 实际对应 reset 后约 0.226 s，而非物理冷启动；此时已有 contact、ctrl 和求解状态。
- matched fork 证明：closed exact state 可精确复现未来 50 ms；由 93D+root 重建的 visible state 会立即分叉并改变接触对。
- 精确 q/qdot 即使清空 time/warmstart 仍可复现，故旧 replay 合同失败的主因是低频 telemetry 采样/重建不等价于 closed post-step state，而非神秘 solver warmstart。

## 2. 获得首条真实 closed-native 完整轨迹

- 唯一 Stage250 straight closed episode：14,406 个 1 kHz physics step、710 条 telemetry，startup/move/stop/full 全通过。
- 位移约 1.180 m，横漂约 0.084 m，stop drift 约 0.067 m。
- 承重核心段 slip 在裁去接触窗首尾 50 ms 后通过 0.10 m/s 门；旧 raw p95 主要被冲击与接触点切换放大。
- 但去抖后稳定 DS→SS→DS 周期为 0，说明它主要是双脚近地 shuffle，不是可靠单支撑步态。
- 因此它可作 X2-native physical warm-start / 速度型保底后端，不是可直接训练的 contact-phase Silver teacher。

## 3. faithful Any2Any 路线的真实边界

- 修复 live trainer 后，WBT29、head2 nominal、Gold/Bronze split、exact-S7 注入范围和 B=0 等价均在真实 Isaac live 环境通过。
- Bronze-only exact-S7 1-update 数值健康且未明显破坏 Gold；最多 5-update 在 U4 因动态 lunge 六项 tracking 全部变差而硬停。
- train mean episode length虽从 11.73 升至 19.15，目标 lunge survival 始终只有 0.22 s，证明 aggregate length 上升不是目标动作进步。
- 结论：Any2Any 的 PPO 管线可以忠实运行，但现有三条 kinematic Bronze 数据不足以产生动态迁移，不允许长训。

## 4. 分层全身遥操作原型已成立但未晋级

- closed Stage250 lower/waist/root 后端保持冻结，仅向 upper14 注入有界 AMASS 上肢目标。
- 原幅值 candidate 全程不倒，startup/move/stop/full 均通过；upper RMSE 改善约 19.1%，p95 改善约 17.9%。
- 代价是横漂约 0.084→0.151 m、heading error 约 0.129→0.205 rad，并恶化部分接触指标。
- 左右镜像后横漂/yaw 不反号，排除简单单侧动量为主因；简单 centroidal reaction feedforward 的量级和相关性也不足。
- half upper 仍不倒并回收部分漂移，但 tracking 与漂移回收均略低于预注册门；停止幅值扫描。
- 准确定位：分层接口可用，但当前 lower backend 的扰动吸收边界不足。

## 5. 已停止的训练/teacher 路线

- recorded-control 局部关节 CEM：严格 matched 修正后，三事件 teacher 最佳仅 26 ms 离地、0.63 mm clearance，0/120 通过；停止该低维 teacher 表示。
- dense upper-robust lower：单 update KL 均值 5.70，A/B survival 从 4 s 降至约 1.7 s；立即否决。
- rank-4 LoRA upper-robust lower：首次 1-update 小幅改善，但 fresh 同 source/seed 重复时 B score 下降 0.00986；效应小于运行波动，停止 LR/rank 扫描。
- 当前最佳仍是冻结 Stage219/Stage250 source，不部署任何新训练权重。

## 6. 版本与大文件归档

- Git 代码/小报告：`75c0d3e`、`989a038`、`0a97a90`、`cb78eda` 已推送。
- 百度 Phase47 归档上传成功，约 300 MB。
- 百度 Phase55 归档上传成功，约 495 MB。
- 按用户约定，上传成功即完成，本轮未做远端二次核验。
- 旧 Stage222–225 文件未触碰、未提交。

## 7. 下一次真正值得做的工作

优先级一：把分层方案作为系统贡献继续，而不是追求当前 Stage250 变成标准抬脚步态。固定 source lower backend，扩大多条 upper clip、速度和转向的独立评估，建立可复现扰动统计；只有效应跨 episode/clip 稳定后再训练鲁棒 adapter。

优先级二：若必须获得标准 locomotion teacher，停止小关节 residual/CEM，改用直接足端/contact 或整段动力学轨迹优化，并把 liftoff/touchdown/contact mode 作为显式变量。

优先级三：等获得更多独立动态 GMR reference 后，再恢复 faithful Any2Any exact-S7；不要用当前三条 Bronze 继续长训。

## 最终阶段裁决

```text
official-domain contradiction: resolved
closed native speed backend: usable as fallback / warm-start
standard single-support locomotion seed: not achieved
hierarchical upper teleoperation: demonstrated, not promoted
faithful Any2Any long training: locked
upper-robust lower long training: stopped
best deployable checkpoint: original Stage219/Stage250 source
```
