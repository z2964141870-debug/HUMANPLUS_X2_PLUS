# SONIC-X2 transfer-v2 独立复现与衣服遥操可用性审计

日期：2026-08-17  
范围：仅 hp3090 离线 MuJoCo / ONNX 分析；未连接 X2、Orin、BLE 或电机。  
任务卡：`/home/yu/projects/AGENT_TASK_CARD.md` v1.1。

## 结论

建议把 `x2_sonic_frozen_g1core_lora_v2.onnx` 升级为下一阶段 X2 遥操的首选
policy，停止继续围绕旧 `x2_sonic_policy.onnx` 或 native-14k 权重做主要适配。
它在相同 PHUMA-96、相同 X2 MuJoCo、相同观测和控制参数下，呈现更低的关节
跟踪误差、根部漂移和关节越界，并且在没有真实未来轨迹、只重复当前参考姿态时
仍能保持这些优势。

但它尚不能被称为“可直接上真机的衣服遥操包”。当前缺口不在 policy 本体，
而在衣服端到作者 ZMQ v5 参考协议的适配、完整 C++ deploy 的 sim2sim parity、
手腕旁路语义，以及 X2 三机部署和安全状态机。论文也明确把 transfer-v2 的真机
验证列为 ongoing。

## 版本与位置

- 快速复现仓库：
  `/media/yu/FAFF-E977/8_16_A2A_x2/sonic-x2`
  - commit：`d2384821583007306f5feee6cffa0ed3e78ab3f8`
  - branch：`main`
  - 工作树：clean
- X2 deployment 稀疏审计副本：
  `/media/yu/FAFF-E977/8_16_A2A_x2/GR00T-WholeBodyControl-X2-review-audit`
  - commit：`70bed4539efae431621013b3df71e6f81df1ab1c`
  - branch：`main`
  - 工作树：clean
  - 仅检出本次需要的 X2 deploy、ZMQ、teleop 和文档文件；未拉取 LFS 大文件。
- 实验代码项目：`/home/yu/projects/BFM-Zero`
- 实验数据根：`/media/yu/FAFF-E977/8_16_A2A_x2`
- Python 环境：`/home/yu/env/x2_teleop_final/sonic_runtime`

模型：

| 模型 | 大小 | SHA256 |
|---|---:|---|
| transfer-v2 | 57,692,064 B | `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9` |
| repo native-14k | 58,505,726 B | `0228147dc5aed08cd3f9388c31eb48b39abbef58c5f5ca39ea04666051de25ae` |
| 前两天实际测试权重 | 58,505,722 B | `e7ccd6522010ea660facfb7265fb129dac2b580dd1483cc1dbada73c205309d7` |

前两天实际权重与仓库 native-14k 并非同一数值模型；不能只按文件名把二者视为
同一基线。

## 方法是否真的是 Any2Any 思路

是，但它利用了 G1 与 X2 极高的骨架相似性，把 Any2Any 的 learned alignment
缩成了解析 codec：

1. 以成对重定向数据拟合逐关节 affine map，把 X2 参考和 proprio 编码到 G1
   语义空间，再把 G1 action 解码回 X2。
2. 冻结三个 encoder、FSQ token bottleneck、kinematic decoder 和 dynamics decoder
   原权重。
3. 只在 dynamics decoder 的线性层加入零初始化 LoRA，约占平台参数的 0.25%，
   用闭环 RL 学习 X2 的动力学差异。
4. 用独立 OOD gate 选择峰值 checkpoint，避免后期训练出现“误差更小、覆盖率更差”
   的过拟合。

这与我们之前反复做 AMASS / PHUMA / BONES-SEED / DDR 重定向的关键差别是：
重定向只解决“目标长什么样”，新方法把 G1 大规模 motion prior 原样保留，并把
可训练自由度严格放在能看到 posture、contact、velocity 的 dynamics decoder 内。
它没有证明“重定向不重要”，而是证明单独重定向不足以跨过动力学差异。

同时，这不是一个廉价魔法。论文给出的 selected lineage 约为 135 GPU-hours、单个
8-GPU 节点 overnight；它比 SONIC 预训练便宜很多，但仍远大于随手在单张 3090
上试几个短 run。其成功还依赖一个很强的前提：G1 与 X2 是 joint-for-joint 近似匹配。

## 官方三条动作复现

使用上游推荐配置：transfer-v2 为 parity gains、`action_clip=20`、冻结手腕；
native-14k 使用其 incumbent-specific `bigrun` preset。

| 动作 | transfer-v2 joint MAE | native-14k joint MAE | 结果 |
|---|---:|---:|---|
| relaxed walk | 0.1172 rad | 0.1779 rad | 两者均 motion_end，v2 更好 |
| Gangnam dance | 0.1873 rad | 0.2152 rad | 两者均 motion_end，v2 更好 |
| idle stand | 0.1555 rad | 0.1964 rad | 两者均 motion_end，v2 更好 |

去掉手腕冻结后，三条动作仍能跑完，但最大关节误差升到约 1.95–2.17 rad。这验证
了论文的 frozen-latent bottleneck：原 SONIC 表征没有可靠携带绝对 wrist pose，
下游 LoRA 也无法恢复它。

## PHUMA-96 同条件对比

数据：
`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/canonical_motion_bank/phuma_stratified96_v4`

配置：50 Hz policy、5 秒/clip、training-parity PD、`action_clip=20`、相同 X2 场景、
相同 wrist reference override、CUDA FP32。这里的 96 条是我们自己的分层 OOD 小样本，
不能与论文的 1,931 条 PHUMA 数字直接横比。

完整 0.9 秒未来参考（oracle）：

| 模型 | pass | 平均存活 | 平均 joint MAE | 平均漂移 | 平均 tilt | 平均越界 |
|---|---:|---:|---:|---:|---:|---:|
| 前两天实际权重 | 50–51/96 | 3.374 s | 0.1187 rad | 0.802 m | 0.573 rad | 0.058 rad |
| repo native-14k | 52/96 | 3.443 s | 0.1187 rad | 0.948 m | 0.565 rad | 0.052 rad |
| transfer-v2 | 54/96 | 3.465 s | 0.1052 rad | 0.597 m | 0.530 rad | 0.033 rad |

旧实际权重的一条 clip 位于 5 秒门的边界，重复运行由 4.8 秒跌倒变为刚好通过，
所以记录为 50–51，而不是把一条边界样本包装成确定提升。更可信的是连续指标：
transfer-v2 相对旧实际权重，joint MAE 约下降 11%，漂移约下降 26%，关节越界约
下降 44%。

这低于论文宣称的 PHUMA `59% -> 69%` 十点提升。原因至少包括样本规模、物理引擎、
重定向和终止规则不同；论文本身也明确报告未能复现 SONIC 作者的 PHUMA 数字。

## 实时衣服的因果未来窗消融

ONNX 输入是 1670 维：680 维参考 tokenizer + 990 维机器人 proprio history。
参考部分包含当前帧和 9 个未来帧，间隔 0.1 秒。衣服在线只能直接提供当前帧，
所以新增了不改变 oracle 默认行为的评估选项：

- `oracle`：完整真实 0.9 秒未来；
- `hold`：十个槽全部重复当前姿态，未来速度置零；
- `lookahead_0p1/0p2/0p3`：只提供对应长度真实未来，剩余槽保持末帧。

transfer-v2 结果：

| 参考可用性 | pass | 平均 joint MAE | 平均漂移 | 平均 tilt |
|---|---:|---:|---:|---:|
| oracle 0.9 s | 54/96 | 0.1052 rad | 0.597 m | 0.530 rad |
| hold 0 s | 54/96 | 0.1030 rad | 0.573 m | 0.543 rad |
| 0.1 s | 56/96 | 0.1029 rad | 0.620 m | 0.524 rad |
| 0.2 s | 55/96 | 0.1043 rad | 0.670 m | 0.523 rad |
| 0.3 s | 55/96 | 0.1053 rad | 0.683 m | 0.526 rad |

结论不是“未来窗没用”，而是：在这套 5 秒 PHUMA 稳定与跟踪门下，简单 current-pose
hold 已足以作为衣服 sim2sim 的第一版 adapter，不必先训练未来预测器。以后如果真实
衣服数据证明快速动作滞后明显，再引入短时运动学外推或 learned planner；不应在尚未
观察到问题前先增加模型复杂度。

相同 hold 条件下：

| 模型 | pass | 平均 joint MAE | 平均漂移 | 平均越界 |
|---|---:|---:|---:|---:|
| 前两天实际权重 | 53/96 | 0.1184 rad | 0.811 m | 0.053 rad |
| repo native-14k | 51/96 | 0.1237 rad | 0.882 m | 0.057 rad |
| transfer-v2 | 54/96 | 0.1030 rad | 0.573 m | 0.035 rad |

因此 v2 的优势在衣服可实现的 causal-hold 输入下仍保留，主要体现为跟踪、漂移和
越界，而不是 pass 数的巨大跃升。

## 衣服接入的正确链路

当前 X2 衣服脚本的末端是 62 维 AIMRT qpos，直接送官方 `teleop_bridge`。这是“直接
位置重定向”链，不经过 SONIC。新方案应改成：

```text
上衣/裤子 IMU
  -> TIC/LFP 人体姿态估计
  -> SMPL/GMR 得到 X2 joint_pos_mj 当前帧
  -> causal reference adapter（v1 先 current-pose hold）
  -> ZMQ v5 pose :5556
       joint_pos_mj (31)
       joint_pos_mj_future (9,31)
       root_quat_xyzw + future
       joint_vel_mj_future
       frame_index / future_dt_s
  -> PC2 pose proxy / watchdog
  -> X2 C++ SONIC deploy（1670 -> 31）
  -> X2 电机接口
```

可复用部分：BLE 接收、T-pose 校准、人体姿态 ONNX、SMPL/GMR、同步/丢包日志。
需要替换的部分：`make_official_x2_qpos -> AIMRT teleop_bridge` 发布端；改为作者的
ZMQ v5 packed pose publisher。机器人 proprio 990 维由 PC2 C++ deploy 从真实状态
构造，不由衣服端伪造。

## 仍需关闭的四个部署风险

1. **deployment commit 早于 transfer-v2 release。** 审计的完整 runtime commit 是
   2026-08-14，transfer bundle commit 是 2026-08-16。runtime 支持通用 `--model`、
   `--action-clip`、ZMQ 和 wrist bypass，但不能假设它已经为 v2 设置了正确默认值。
2. **手腕语义不完全一致。** quick-play 的 `--freeze-wrist` 冻结 6 个 wrist DOF；C++
   runtime 的 `--wrist-bypass=ik` 只旁路 pitch/roll 四轴，wrist yaw 仍由 SONIC 控制。
   必须在完整 C++ sim2sim 中做逐轴 parity 后再确定最终配置。
3. **论文的 v2 真机证据尚未完成。** 原生 SONIC-X2 家族有真机案例，但论文 limitation
   明确说 transferred controller 的 physical verification ongoing，不能偷换成“v2 已真机验证”。
4. **衣服噪声尚未评估。** 当前消融使用干净 PHUMA 当前帧；衣服的漂移、抖动、掉帧、
   T-pose 偏差和 35–60 Hz 抖动必须用后续真实记录回放验证。

## 下一阶段 gate

只在 hp3090 / 官方 X2 MuJoCo 上完成以下 gate，不接机器人：

1. 把衣服当前 `qpos` publisher 改成 ZMQ v5 publisher，先支持 current-pose hold。
2. 启动作者完整 C++ X2 deploy 的 sim 模式，显式传入 transfer-v2 模型、
   `action_clip=20`、不加载 incumbent `bigrun` preset。
3. 比较 Python quick-play 与 C++ deploy 的首帧观测、首帧 action、关节顺序、默认角、
   action scale、PD 和 wrist bypass。
4. 用离线 PHUMA 流回放，要求无异常跳变；再用真实衣服日志回放，扫描延迟、掉帧和
   噪声门。
5. 只有 C++ sim2sim、故障回退和 Ctrl+C/断流安全状态全部通过，才生成 X2 三机部署
   包和真机 dry-run 清单。

## 最终判断

这个新版本有价值，而且比前两天的两个 SONIC-X2 权重更适合作为衣服遥操底座。
它第一次给我们的 Any2Any 方向提供了可运行的 X2 权重、codec sidecar 和完整 deploy
接口，而不是只给论文思路。但它解决的是“policy 与跨具身动力学”这一层，不会自动
解决衣服噪声、实时协议、手腕旁路和 X2 三机安全切换。项目应从“继续训练新 policy”
转为“复用 transfer-v2，完成衣服 ZMQ adapter + C++ sim2sim deployment closure”。

