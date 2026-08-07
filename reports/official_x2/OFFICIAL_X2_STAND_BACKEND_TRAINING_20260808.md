# X2 Official-MuJoCo Stand/Stop Backend：阶段记录（2026-08-08）

## 结论

独立 stand actor 已在 IsaacLab 达到 10 秒稳定站立，但单独迁入官方 AimDK v1.0 MuJoCo 会在约 2 秒后倒地。将它与官方域中稳定的 Stage208 actor 按动作空间 `50% / 50%` 混合后，得到首个可复现的官方域站立与停车吸引域：两次纯站立均连续 10 秒不倒、XY 漂移约 2 cm；完整 `stand→walk→stop` 中站立门和停车门通过，当前唯一未闭合的是行走航向与横漂。

## 假设 / 干预 / 对照 / 结果 / 结论

### 1. 独立站立策略

- **假设：** Stage208 零命令下仍带强行走先验，单独训练小型 stand skill 比让它原地遗忘更容易。
- **干预：** 93D observation、15D 腰腿 action contract 不变，从零训练 stand backend；脚踝 PD 匹配官方 Kp/Kd=40/20。
- **对照：** Stage208、Stage208→stand i50、balance i50、scratch i100/i150。
- **结果：** scratch i150 在 IsaacLab 确定性回放中三次达到 10 秒，位移 `-1.79/+1.33 cm`、最低 root-z `0.629 m`、最大倾角 `0.0956 rad`、动作饱和 `0%`、双支撑 `100%`。
- **结论：** IsaacLab 站立技能成立，但不同 seed 的 PLAY reset 实际相同，只证明确定性可复现，不证明扰动鲁棒性。

### 2. 官方 MuJoCo 单策略与交接

- **假设：** i150 可直接迁移，或经 Stage208 热启动后进入其吸引域。
- **干预：** 官方 AimDK v1.0 MuJoCo、50 Hz、官方 ankle Kp=40、Stage208 默认姿态。
- **对照：** i150 直接启动；Stage208 3 秒热启动后切 i150。
- **结果：** 直接启动最终 root-z `0.091 m`；热启动后约 1.84 秒倒地，最终 root-z `0.095 m`。
- **结论：** 不是 ONNX 导出误差或单纯开局姿态问题，而是 i150 缺少官方域稳定反馈。

### 3. 有界双策略混合

- **假设：** Stage208 提供官方域平衡反馈，stand actor 抑制其零命令行走先验，两者存在可用混合区间。
- **干预：** 两个 actor 分别维护 93D observation 中的 last-action 状态；最终 15D raw action 做有界线性混合。
- **对照：** stand 比例 25%、50%、100%；有/无 3 秒热启动。
- **结果：**

| 官方 MuJoCo 条件 | 10 s 生存 | XY 漂移 | 最低 z | 最大倾角 | 末 1 s 速度 |
|---|---:|---:|---:|---:|---:|
| 100% stand，直接 | 否 | 0.730 m | 0.091 m | 1.523 rad | 0.0004 m/s |
| Stage208 3 s → 100% stand | 否 | 0.300 m | 0.095 m | 1.528 rad | 0.0010 m/s |
| Stage208 3 s → 25% stand | 是 | 0.909 m（热启动占 0.734 m） | 0.601 m | 0.229 rad | 0.0062 m/s |
| Stage208 3 s → 50% stand | 是 | 0.830 m（热启动占 0.760 m） | 0.613 m | 0.230 rad | 0.0229 m/s |
| **直接 50% stand，run 1** | **是** | **0.0212 m** | **0.6447 m** | **0.1364 rad** | **0.00203 m/s** |
| **直接 50% stand，run 2** | **是** | **0.0202 m** | **0.6456 m** | **0.1352 rad** | **0.00200 m/s** |

- **结论：** 50% 是当前官方域 stand sweet spot；混合不是事后掩盖倒地，而是两个互补闭环形成的新稳定吸引域。

### 4. stand→walk→stop 闭环

- **干预：** 2 秒 50% 混合站立 → 6 秒 Stage208 `vx=0.30 m/s` → 8 秒 50% 混合停车。
- **统一门禁：** stand 漂移≤0.10 m、末速≤0.03 m/s；move 前进≥0.50 m、横漂≤0.30 m、航向≤0.30 rad；stop 漂移≤0.15 m、末速≤0.03 m/s，且各段满足高度/倾角限制。
- **结果：**

| 子门 | 关键数值 | 结果 |
|---|---|---|
| stand | 漂移 0.016 m，末速 0.0116 m/s，倾角 0.138 rad | **PASS** |
| move 生存/前进 | 前进 1.084 m，最低 z 0.652 m，倾角 0.319 rad | **PASS** |
| move 航向/横漂 | 横漂 -0.672 m，航向峰值 0.479 rad | **FAIL** |
| stop | 0.98 s 内持续低于 0.03 m/s；总漂移 0.0446 m；末速 0.0034 m/s | **PASS** |
| full | move 航向/横漂未过 | **FAIL** |

## 已排除路线

- 纯 PD 保持默认姿态：官方 MuJoCo 中倒地。
- 只提高 heading command 外环增益：非单调且会恶化偏航。
- 翻转 heading gain：同样无改善，Stage208 对 `wz` 输入权威过弱。
- 单独 hip-yaw feedback：只小幅改善 6 秒航向，横向位移反而增加，不能单独晋级。
- 继续强迫 Stage208 学静止：短微调仍保留约 46% action 饱和，独立技能结构更干净。

## 当前 checkpoint 与资产

- Stage208 locomotion：`stage208_s2550_actor.onnx`
- Stand backend：`stand_backend_scratch_i150_actor.onnx`
- IsaacLab checkpoint：`logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_01-17-25_stage_stand_backend_scratch_resume100_to300_v1/model_150.pt`
- 当前官方统一结果：`gate_v1_walk030_6s_stop_blend050_8s.json`

## 下一步

固定已经通过的 50% stand/stop，不再扫它。只针对 Stage208 行走段的 yaw/lateral 协调做低维、官方域可验证的修正；若低维 teacher 不能把横漂从约 0.67 m 压到 0.30 m 内，则进入官方模型参数匹配的 IsaacLab 专项微调，而不是继续扫 heading gain。
