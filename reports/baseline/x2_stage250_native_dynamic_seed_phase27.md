# BASE Phase27：Stage250 X2-native dynamic-seed 资格审计

## 一句话裁决

Stage250 的直行、右转、左转三条 official full-gate trace 都能作为 **X2 原生运动学/控制 warm-start**，但都不能直接作为接触一致的动态 Silver seed，更不能把离线估计的 GRF 当作动力学真值；正确用法是“拿它们初始化 X2 动力学优化，再重新求 contact/root/force”，而不是原样训练。

## 游戏任务

- [x] 只读使用三条各 710 行 Stage250 official trace；没有新 rollout。
- [x] 显式解码 93D observation：`3+3+3+3+31+31+15+4`。
- [x] 核对 Stage219/stand ONNX 的 `93→15` I/O、31D joint order、15D action order、default、scale 与 action contract。
- [x] 用 yaw + projected gravity 重建完整 root quaternion，并逐帧反校验。
- [x] 在未修改的官方 `x2.xml` 上重建 FK、sole collision、COM、foot trajectory 与 centroidal quantities。
- [x] 用 50 Hz finite-difference acceleration、重建 PD torque、`mj_rne` 和摩擦金字塔 NNLS 估计 GRF。
- [x] 明确 GRF/contact 是模型估计，不是 closed ROS、足底传感器或真机真值。
- [ ] 接触一致动态 seed 资格。
- [ ] 动力学真值 seed 资格。

## 假设

Stage250 三条轨迹来自目标 X2 官方 MuJoCo 且均通过当时 full gate，可能比 AMASS/GMR 更适合作为 X2-native lower-body teacher 的初始轨迹。但 full gate 只证明存活、移动、转向和停止，不自动证明足端无滑移、接触时序正确或动力学残差闭合。

## 干预

本阶段无控制干预。对 trace 做以下确定性离线变换：

```text
93D obs + root xyz/yaw
        ↓
full qpos/qvel + root quaternion
        ↓
official X2 FK / COM / sole collision / centroidal state
        ↓
50 Hz qacc + reconstructed PD + mj_rne
        ↓
friction-pyramid NNLS foot-GRF model estimate
```

root 没有被 teleport 或优化，joint/root 原始序列没有改写。

## 对照与合同核查

### 93D observation

| slice | 维度 | 语义 |
|---|---:|---|
| 0:3 | 3 | body-frame base linear velocity |
| 3:6 | 3 | torso-IMU angular velocity |
| 6:9 | 3 | projected gravity |
| 9:12 | 3 | command vx/vy/wz |
| 12:43 | 31 | `q - default`，Isaac joint order |
| 43:74 | 31 | dq，同一 joint order |
| 74:89 | 15 | previous clipped actor residual |
| 89:93 | 4 | sin/cos phase + 左右 generator contact intent |

三条均为 `prepare=10 + decoded=700` 行；decoded 行全部是 93D obs 与 15D executed action。root quaternion 重建后 projected-gravity 最大误差为 `3.1e-8`，yaw 最大误差为 `1.7e-16 rad`，31 个 joint 无限位越界。

### action

合同与 Stage250 固结论一致：

```text
actor mean
→ clip[-1,1]
→ + 0.15 gait template / action_scale
→ execution clip[-1,1]
→ target = default + action × LOWER_SCALE
```

Stage219 main actor 与 stand actor 的 ONNX I/O 均核对为 `93→15`；精确路径和 SHA256 在机器可读 JSON 中。

## 结果

### 1. 状态连续性足以作为 warm-start

| trace | joint FD-vs-dq p95 | root linear p95 | root angular p95 | COM FD-vs-model p95 |
|---|---:|---:|---:|---:|
| straight | 0.183 rad/s | 0.028 m/s | 0.238 rad/s | 0.030 m/s |
| turn right | 0.198 rad/s | 0.033 m/s | 0.254 rad/s | 0.033 m/s |
| turn left | 0.183 rad/s | 0.123 m/s | 0.833 rad/s | 0.115 m/s |

下肢 joint q/dq 连续性三条都合格，故可以作为优化初值。左转的 root/IMU 连续性明显差于另外两条，说明 50 Hz 不同 topic 的时间/坐标一致性不够好；它可以保留为 warm-start，但不能当高置信动力学标签。

### 2. 确实包含目标机器人原生移动与转向语义

| trace | move yaw | forward | lateral | COM speed mean |
|---|---:|---:|---:|---:|
| straight | +0.072 rad | 1.204 m | -0.099 m | 0.305 m/s |
| turn right | +0.338 rad | 1.301 m | +0.142 m | 0.338 m/s |
| turn left | -0.312 rad | 1.259 m | -0.097 m | 0.322 m/s |

三条 yaw 与既有 summary 逐项一致。它们不是“静态站立伪装成步态”，而是真实完成了目标机器人位移/转向的 closed-loop rollout。

### 3. 接触周期存在，但不是干净动态参考

generator intent 每条都包含 10 个 DS→SS→DS 周期；但 official sole collision FK 显示摆脚非常低、支撑阶段存在明显瞬时滑移：

| trace | L/R generator-stance slip p50 | L/R slip p95 | L/R swing clearance p95 |
|---|---:|---:|---:|
| straight | 0.036 / 0.052 m/s | 1.116 / 1.406 m/s | 0.0069 / 0.0083 m |
| turn right | 0.039 / 0.070 m/s | 1.239 / 1.428 m/s | 0.0074 / 0.0095 m |
| turn left | 0.158 / 0.123 m/s | 1.102 / 1.377 m/s | 0.0062 / 0.0077 m |

中位 slip 在直行/右转的多数支撑帧较低，但 p95 超过 1 m/s，且摆脚 p95 clearance 只有约 6–10 mm。这更像低抬脚、切换期刮擦/滑动的速度型 rollout，不是可直接监督“可靠卸载—离地—落脚”的动态 reference。

摩擦金字塔从 16 个非负 ray 拟合 6 个 root 方程，是欠定估计；raw force contact 会产生 28 次快速切换，不能把它冒充实际接触标签。generator 的 10 个周期只代表 controller intent，也不能冒充足底测量。

### 4. 动力学闭合只支持“模型可解释”，不支持“真值”

| trace | full generalized residual p50 | p95 |
|---|---:|---:|
| straight | 0.157 | 0.276 |
| turn right | 0.163 | 0.291 |
| turn left | 0.505 | 0.621 |

root 六维方程可由欠定 contact rays 低残差拟合，但 whole-body residual 尤其左转仍大。主要不可观测量是 1 kHz qacc、实际饱和后 torque 和物理子步 contact impulse；因此这些数值只能做模型诊断，不能成为 GRF/COP/centroidal truth label。

### 5. 持续后仰仍被保留

| trace | move signed pelvis pitch mean |
|---|---:|
| straight | -0.197 rad（-11.29°） |
| turn right | -0.178 rad（-10.20°） |
| turn left | -0.187 rad（-10.72°） |

负值为后仰。这与 Phase7 结论一致：Stage250 的位移能力是真的，但自然姿态仍有系统性偏差。因此可把轨迹当 warm-start，却不能把这套 root/hip sagittal 平衡点原样蒸馏为理想风格。

## 结论

### 可做什么

三条全部晋级为：

> **X2-native kinematic/control warm-start seed**

它们可用于初始化后续 direct-official-MJCF 的接触感知轨迹优化、bridge teacher 或 shooting/CEM；比从人体运动学 reference 猜一条 X2 步态更接近目标本体闭环。

### 不能做什么

三条全部不晋级为：

- contact-consistent dynamic Silver seed；
- measured GRF/COP dataset；
- dynamics ground-truth teacher；
- 自然姿态 locomotion reference。

因此 Phase27 的收获不是“终于有三条完美动力学数据”，而是确认我们已有三条**可优化的原生初值**。这能绕开 AMASS→X2 的部分运动学错配，但不能绕开接触修正和动力学优化本身。

## 最小补录字段

若以后允许扩展 Stage250 recorder，想把它提升为高置信动态 seed，至少需要：

1. 1 kHz simulator qpos/qvel 或精确逐样本 timestamp，用于可信 qacc；
2. 31D 饱和后实际 actuator torque；
3. 每物理子步 contact geom identity、position、impulse/force；
4. 明确 frame 的完整 root quaternion、linear/angular velocity；
5. clip 前后 joint target 与逐关节 Kp/Kd。

现有 trace 已足够做 warm-start，故本阶段不擅自新跑。

## 下一步

若主线采用这些数据，唯一合理用法是：以三条 Stage250 q/root/action 作为初值，在 direct official MJCF 中重新优化低维 contact/root/foot trajectory，并把“低滑移、真实离地、后仰减小、原有 forward/turn 语义不退化”作为硬门；在得到优化后 physical rollout 前，不应拿原 trace 扩大训练。

## 证据与边界

- 机器可读结果：`reports/official_x2/phase27_stage250_native_dynamic_seed_audit.json`
- 工具：`tools/official_x2/audit_stage250_native_dynamic_seed.py`
- 测试：`tests/test_phase27_native_dynamic_seed.py`（4 passed）
- 未训练、未新跑 official、未改 adapter、未改 Stage250、未碰 WBT/Git/百度/真机。
