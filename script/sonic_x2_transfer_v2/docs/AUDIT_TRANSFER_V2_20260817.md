# SONIC-X2 transfer-v2 独立复现与衣服遥操可用性审计 (2026-08-17)

来源: hp3090 审计报告 (任务卡 AGENT_TASK_CARD.md v1.1)

## 结论
- **x2_sonic_frozen_g1core_lora_v2.onnx** 升级为下一阶段 X2 遥操首选 policy
- 相对旧权重: joint MAE ↓11%, 漂移 ↓26%, 关节越界 ↓44% (PHUMA-96 同条件)
- causal-hold (当前帧重复) 下优势保留: pass 54/96, MAE 0.1030, 漂移 0.573m
- 尚不能直接上真机: 衣服 ZMQ v5 适配 / C++ sim2sim parity / 手腕旁路 / 三机安全未关闭

## 权重
- transfer-v2: 57,692,064 B, SHA256 8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9
- native-14k: 58,505,726 B (仓库) / 58,505,722 B (前两天实测, 不同数值模型)

## 方法
G1/X2 骨架近似匹配 -> 解析 codec: X2 参考/proprio 编码到 G1 语义空间,
冻结 encoder/bottleneck/decoders, 仅 dynamics decoder 线性层加 0.25% LoRA
闭环 RL 学动力学差异. 论文 135 GPU-hours 训练成本.

## 官方三条动作复现 (transfer-v2 均更好)
- relaxed walk: 0.1172 vs 0.1779 rad
- Gangnam: 0.1873 vs 0.2152 rad
- idle stand: 0.1555 vs 0.1964 rad
- 去手腕冻结 -> 最大误差 1.95-2.17 rad (frozen-latent 不携带绝对 wrist)

## 衣服接入正确链路
上衣/裤子 IMU -> TIC/LFP -> SMPL/GMR X2 joint_pos 当前帧
-> causal reference adapter (v1: current-pose hold)
-> ZMQ v5 pose :5556 (joint_pos_mj 31 + future 9x31 + root_quat_xyzw + vel + frame_index)
-> PC2 pose proxy / watchdog -> X2 C++ SONIC deploy (1670->31) -> 电机

## 仍需关闭的部署风险
1. deployment commit (08-14) 早于 transfer-v2 release (08-16)
2. 手腕语义: quick-play --freeze-wrist 冻结 6 DOF vs C++ --wrist-bypass=ik 只旁路 4 轴
3. 论文 v2 真机验证 ongoing
4. 衣服噪声 (漂移/抖动/掉帧/35-60Hz 抖动) 未评估

## 下一阶段 gate (hp3090, 不接机器人)
1. 衣服 qpos publisher 改 ZMQ v5 (current-pose hold)
2. C++ X2 deploy sim 模式 + transfer-v2 显式参数 (action_clip=20, 不加载 bigrun preset)
3. Python quick-play vs C++ deploy 首帧观测/action/关节序/PD/wrist parity
4. 离线 PHUMA 回放 + 真衣服日志回放 (延迟/掉帧/噪声门)
5. C++ sim2sim + 回退 + 安全状态全过 -> 三机部署包 + 真机 dry-run 清单
