# X2 官方 MuJoCo × Stage208 无训练验证（2026-08-07）

## 一句话结论

官方 X2 v1.0 MuJoCo/ROS/ONNX 部署链已经打通，SDK 自带 ONNX 严格启动后可稳定运行；但当前 Stage208 在 IsaacLab 中成功的前进能力不能直接迁入官方 MuJoCo。失败已被定位为 Stage208 的 sim-to-sim 泛化/执行动力学问题，而不是官方环境不可用、ONNX 导出、29/31DOF 解释或步态模板本身。

## 已确认的接口

- 官方模型为 X2 31DOF；样例 policy 控制 29DOF body，仅锁定 2 个头部关节。
- SDK 自带舞蹈 ONNX 可由官方 controller 正常加载：`151D obs + time_step -> 29D action`。
- Stage208 checkpoint 已无损导出为 `93D obs -> 15D 腿腰 residual` ONNX；PyTorch/ONNX 最大绝对误差为 `9.54e-7`。
- Stage208 的 31DOF 观测顺序、15DOF action 顺序、默认姿态、50Hz 控制周期、PD 目标发布和 gait template 均已接入官方 ROS topic。

## 关键实验

| 对照 | 8 秒结果 | 首次失败 | root z min | tilt max | XY 位移 | 解释 |
|---|---:|---:|---:|---:|---:|---|
| 官方舞蹈 ONNX，direct RL，60 秒 | 通过 | 无 | 0.511 m | 0.594 rad | 0.242 m | 完整配置时长覆盖，官方仿真、controller 与 ONNX 链有效 |
| 仅 gait template，vx=0.30 | 通过 | 无 | 0.647 m | 0.053 rad | 0.016 m | 模板安全，但单独几乎不走 |
| Stage208 full，vx=0.30 | 失败 | 1.56 s | 0.166 m | 1.504 rad | 0.486 m | actor 闭环在官方域失稳 |
| Stage208 full，vx=0.20 | 失败 | 1.58 s | 0.163 m | 1.499 rad | 0.504 m | 降速没有解决失稳 |
| IsaacLab 成功关节目标开环回放 | 失败 | 1.02 s | 0.094 m | 1.594 rad | 0.575 m | 绕过 ONNX 后仍失败，物理/执行差异已成立 |

## 排除的伪原因

1. **“29DOF 不是 X2”**：错误。官方本体是 31DOF，官方 policy 与我们的迁移边界一样，只控制 29DOF body。
2. **ONNX 数值导出错误**：已由逐元素等价检查排除。
3. **关节顺序错位**：官方 31DOF 状态与 Stage208 的 IsaacLab 顺序已显式重排；15DOF action 顺序也做了 fail-closed 校验。
4. **长插值先把机器人推倒**：改成官方模型直接以 Stage208 默认姿态初始化后仍失败。
5. **单纯速度过高**：0.20 和 0.30 m/s 的失败时间几乎一致。
6. **模板本身导致摔倒**：模板单独 8 秒稳定。
7. **适配器提前裁剪 actor**：已按 IsaacLab 源码修正为“raw action + template 后整体裁剪”，上一动作观测也保留 raw action；修正后结论不变。
8. **官方 odom 速度坐标错误**：已验证 odom linear velocity 为世界系，并在进入 Stage208 前旋转到 pelvis/body 系；修正后仍失败。
9. **漏用官方启动器或环境变量**：使用 `start_sim.sh -s` 与直接二进制结果一致。

## 官方仿真是否更可信

官方 MuJoCo 应作为当前厂商定义的主评估域，因为它直接提供官方 MJCF、碰撞体、惯量、关节/力矩限制和 SDK 命令语义。严格复现还发现 README 中 `JOINT_DEFAULT` 是可选步骤：当前包的该准备姿态会先失稳；跳过它、让官方 ONNX 从原始 reset 第一帧进入 `RL_DEFAULT` 后，60 秒门禁通过。这证明官方仿真与样例 policy 链本身有效。

官方 MJCF 与自建 sole12 URDF 的足底 12 点几何和主要 link 惯量基本同源。当前最值得进一步核对的差异是：

- MuJoCo `armature=0.03`、`frictionloss=0.3` 与 IsaacLab 分组 armature；
- 官方 MuJoCo 的非相邻 link 自碰撞与 IsaacLab `self_collisions=off`；
- MuJoCo/PhysX 接触求解、摩擦和 mesh collision 行为；
- 官方 command 层 PD/力矩饱和与 IsaacLab implicit actuator 的逐步一致性；
- odometry linear velocity 为世界系，而 Stage208 期望 body 系（它不是开环回放失败的原因，但会影响闭环性能）。

## 当前裁决

- **官方部署链与官方样例：通过。** direct-RL 60 秒内未倒，覆盖配置给出的约 56.4 秒动作长度，官方域可作为新主门禁。
- **Stage208 官方域站立/模板安全性：局部通过。** 模板可稳定，但无有效 locomotion。
- **Stage208 官方域 locomotion：失败。** 使用官方原始 reset、官方启动器、精确 action 递归语义和正确 body 速度后仍倒，当前不具备直接部署或 sim-to-sim 鲁棒性。
- **PD 单变量复查取得局部突破。** 只把踝 Kp 从 300 匹配到官方的 40 后，Stage208 首次在官方域完成 8 秒不倒；但累计偏航约 `-1.16 rad`，停止后仍运动，因此当前失败已从“开局倒地”收窄为“航向与停止闭环错误”。
- **是否推翻先前 Any2Any 工作：否。** 结果证明自建 IsaacLab 的成功不能单域作最终证据，也证明长期卡点不只是 reference 数据；模拟器/执行器域差异同样是硬瓶颈。

## 下一步甜点位

在不训练的前提下，继续扫 reward 或速度价值很低。最有信息增益的下一步是做一个小型 actuator/contact matched audit：让 IsaacLab 临时匹配官方 MJCF 的 armature、joint friction、自碰撞与官方 PD/饱和，逐项回放同一成功 target trace。找到使失败时间显著变化的单一物理项后，再决定是把官方 MJCF 转入训练环境，还是把该差异纳入四域随机化。

## 资产管理

- 代码与本报告进入 `CWI_CrossEmbodiment_Sim` Git 工作树。
- 官方 SDK、Docker 镜像、ONNX 和原始多 MB trace 不进入 Git。
- 原始 trace 与官方 teacher 统一保存在：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807/`。
