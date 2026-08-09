# BASE Phase7：X2 持续后仰的 signed sagittal posture 审计

日期：2026-08-09
范围：只读 trace/官方资产审计与纯测试；无训练、无真机、无 WBT 修改、无 Git/网盘操作、无新增 MuJoCo 运行。

## 结论先行

用户在视频中观察到的“持续后仰”是真实、稳定且跨动作存在的，不是渲染视角造成的错觉。Stage250 三条动作在巡航段的 pelvis/root signed pitch 分别为 **-11.65°、-10.99°、-10.27°**（负值定义为后仰）；Stage351 虽改善到 **-10.14°**，仍是同一失败模式。

现有成对证据排除了两个简单解释：

1. 将 `stage208` default pose 换为 `official_v1`，站立 root pitch 仅从 **-5.20° 改善到 -4.36°**，不足以解释约 10–12° 的动态后仰；
2. 将 waist pitch issued target 缩放从 0.75 降到 0.50，完整运动段 root pitch 只从 **-10.593° 到 -10.577°**，几乎不变。因此腰关节动作会改变局部形状，但不是 pelvis 后仰的主因。

把同一 Stage208 actor 直接换成官方 native PD 会倒地，而官方随包 `kuailechongbai.onnx + 官方模型/PD` 的稳定 60 s trace 的 root pitch 均值为 **+12.63°**。这共同说明：**后仰不是 X2 官方 MuJoCo 本体的必然默认姿态，也不是单个 waist residual 或 default pose 能解释；它主要是迁移 actor 的全下肢 sagittal target 与当前稳定化 PD/default/controller 共同形成的动态平衡点。**

当前仍无官方 locomotion 自然姿态基准，故本阶段不事后发明绝对“自然姿态阈值”，只建立 signed metric 与相对晋升门。

## 1. 预注册假设

以下假设在本阶段不新增仿真运行前锁定；既有 trace 用于零训练裁决。

### H1：default pose 是主因

- 假设：Stage208 的腿部默认姿态与官方 X2 default 不同，造成稳定平衡点后移。
- 预测：只改 `default_pose_profile=stage208 → official_v1`，应大幅且持续地把 signed root pitch 推向非负方向。
- 裁决：**不是主因，存在次要贡献。** 严格单变量站立 A/B 只改善 +0.84°，且仍持续为负。

### H2：waist pitch actor residual 是主因

- 假设：actor 发出的负 waist pitch target 直接造成整体后仰。
- 预测：只降低 waist tilt action multiplier，root pitch 应在 start/move/stop 全阶段近似同比改善。
- 裁决：**不是 pelvis 后仰主因。** 0.75→0.50 使完整运动段 waist issued target 改善约 +1.45°，但 root pitch 只改善 +0.016°；巡航子段甚至恶化 -0.29°。

### H3：官方 PD/domain 是主因

- 假设：当前非官方的 stiff PD 直接把正确 actor target 扭成后仰。
- 预测：同一 actor 只切换到官方 native PD 后应更自然且保持稳定。
- 裁决：**“单换 PD 可修复”被否定；actor-controller contract 不匹配得到支持。** 同一 Stage208 actor 在 `official_native` PD 下倒地，巡航 pitch 坍塌至 -73.29°；而 shipped actor 与其原生 controller 在同一官方物理模型中可以稳定运行且整体为正 pitch。

## 2. 权威量定义

### 2.1 pelvis/root signed pitch

adapter 的 `obs[6:9]` 是 `R^T [0,0,-1]`。采用与 observation 完全一致的计算：

```text
pitch = asin(projected_gravity_x)
```

- 负值：pelvis/root 向后仰；
- 正值：pelvis/root 向前倾；
- 这不是 unsigned `root_tilt`，因此不会把前倾和后仰混为一谈。

### 2.2 waist pitch actual

```text
waist_actual = selected_default_waist + obs_joint_position_residual
```

当前 `stage208` 与 `official_v1` 的 default waist pitch 都是 0°。

### 2.3 waist/lower-body issued target

```text
issued_target = selected_default + trace_action * action_scale
```

trace 中的 `action` 是最终发给 PD 的 normalized action，可能已经经过 template、clip、supervisor 或 controller handoff。报告严格称它为 **issued target**，不伪称为未经修改的 raw actor output。

### 2.4 分阶段规则

- `stand`：trace 原生 stand 标签；
- `start`：move 开始后的预注册启动窗；matched-event 使用记录的 1 s accel，旧 step-command 固定使用首 1 s；
- `move`：启动窗之后的剩余 move；
- `stop`：trace 原生 stop 标签；
- `prepare` 排除。

## 3. 主要结果

### 3.1 Stage250 三动作：后仰跨动作一致

| trace | stand root pitch | start | move | stop |
|---|---:|---:|---:|---:|
| straight | -6.87° | -10.20° | -11.65° | -7.40° |
| turn left | -6.85° | -9.92° | -10.99° | -7.37° |
| turn right | -6.85° | -9.99° | -10.27° | -7.24° |

三条动作的 stand 几乎重合，启动后共同增加到约 -10° 至 -12°，停止后又回到约 -7°。这说明后仰与 locomotion actor/controller 状态强相关，而非某一个转向动作或单次随机漂移。

### 3.2 Stage351：性能主线改善后仍保留 sagittal 失败模式

| phase | root pitch | waist actual | waist issued target | actual-target error |
|---|---:|---:|---:|---:|
| stand | -6.37° | -2.04° | -1.42° | -0.62° |
| start | -4.90° | +0.77° | +0.69° | +0.08° |
| move | -10.14° | -6.67° | -5.08° | -1.59° |
| stop | -6.86° | -1.66° | -0.95° | -0.71° |

Stage351 启动窗一度更直，但进入持续 move 后仍回到约 -10°。因此后仰不是起步瞬态，而是巡航平衡点。

Stage356 symmetry projection 已被上一阶段判定倒地。本报告保留其 trace 作为负对照，但 **不使用其 move/stop 的 -28°/-83° 来判断自然姿态**，因为这些数值已被跌倒污染。

### 3.3 default pose 严格 A/B

既有 `stand-policy-stage208-8s` 与 `stand-policy-official-v1-8s` 的配置唯一差异是 `default_pose_profile`。

| profile | stand root pitch | waist actual | waist issued target |
|---|---:|---:|---:|
| stage208 | -5.20° | +2.80° | +2.27° |
| official_v1 | -4.36° | +4.77° | +3.97° |
| 差值 | +0.84° | +1.97° | +1.70° |

official default 有小幅改善，但量级远小于 Stage250/351 的动态后仰。这项 A/B 的 profile 同时改变腿部与上肢默认值，不能进一步把 +0.84° 归因到某个单关节；但足以否定“default profile 单独就是主因”。

### 3.4 waist issued target 严格 A/B

既有两条 Stage220 straight trace 的配置唯一差异是 `waist_tilt_action_multiplier=0.75/0.50`。

| phase | root pitch @0.75 | root pitch @0.50 | 改变量（正=少后仰） |
|---|---:|---:|---:|
| stand | -6.64° | -6.46° | +0.19° |
| start | -10.05° | -9.12° | +0.93° |
| move | -10.77° | -11.06° | -0.29° |
| stop | -7.16° | -6.84° | +0.32° |
| 完整 start+move | -10.593° | -10.577° | +0.016° |

完整运动段 issued waist target 从约 -4.57° 改到 -3.12°，pelvis pitch 却几乎不动。这是“继续单扫 waist multiplier”信息收益很低的直接证据。

### 3.5 PD 严格 A/B

既有 `stage208_pd_official_kp_ankle_strict` 与 `stage208_pd_official_native_strict` 的配置唯一差异是 PD profile。

| PD | height gate | start root pitch | move root pitch |
|---|---:|---:|---:|
| 当前稳定用 `official_kp_ankle` | pass | -12.96° | -14.63° |
| `official_native` | fail | -15.02° | -73.29°（倒地污染） |

这并不证明 stiff PD 是正确的自然姿态控制器；它只证明官方 PD 不能脱离配套 actor/action/default contract 单独替换。

### 3.6 官方 shipped policy 的独立参照

`official_native_dance_teacher_50hz_60s.npz` 来自 AimDK X2 v1.0 官方 MuJoCo 与随包 `kuailechongbai.onnx`：

- 3000 帧，约 60 s；
- pelvis/root pitch：均值 +12.63°，p05 +2.57°，p95 +24.09°；
- torso IMU pitch：均值 +12.63°；
- waist actual：均值 -3.20°；
- waist command：均值 -7.69°。

它是舞蹈而不是 locomotion，不能拿 +12.63° 当行走自然姿态门槛。但它能排除一个关键误解：**官方 X2 模型/重力/关节定义并不要求 root 长期为负 pitch；负 root pitch 也不是 waist command 为负的必然结果。**

## 4. 下肢 sagittal contract 证据

Stage250 straight 巡航时，最终 issued target 与 actual 已形成稳定但非官方原生的下肢构型：

| joint | default | actual mean | issued target mean |
|---|---:|---:|---:|
| left hip pitch | -14.21° | +10.27° | +6.46° |
| right hip pitch | -14.21° | +0.07° | +0.67° |
| left knee | +30.38° | +14.78° | +11.73° |
| right knee | +30.38° | +17.84° | +15.46° |
| left ankle pitch | -16.17° | -12.72° | -17.13° |
| right ankle pitch | -16.17° | -7.19° | -17.49° |
| waist pitch | 0.00° | -7.79° | -5.92° |

Stage351 的相同表型仍存在（move root -10.14°，waist target -5.08°，左右 hip/ankle target 仍高度不对称）。与 shipped dance 的 joint command 分布不能做动作级一一比较，但二者差异足以提示：下一步应审计 **全 sagittal lower-body target contract 与其相位关系**，而非继续孤立地调 waist。

## 5. 假设—干预—对照—结果—结论—下一步

### 假设

持续后仰主要来自迁移 actor 下肢 sagittal target 与当前官方域稳定化 controller 的联合平衡点；default pose 与 waist residual 只提供次要偏置。

### 干预

本阶段没有新增仿真干预。只建立权威 signed pitch 计算器，并复用三组已有严格单变量 A/B：default profile、waist target scale、PD profile。

### 对照

- Stage250 straight/left/right；
- Stage351 当前 matched-event control；
- Stage356 已拒绝 symmetry 负对照；
- 官方 shipped actor 的 60 s 官方 MuJoCo 稳定 trace。

### 结果

- 后仰跨动作、跨 Stage250/351 稳定存在；
- official default 只改善站立 +0.84°；
- waist target 0.75→0.50 对完整运动段只改善 +0.016°；
- official native PD 与迁移 actor 不兼容并倒地；
- 官方 shipped actor 在同一官方模型中没有负 pitch 必然性。

### 结论

本阶段把“机器人看起来歪”收敛为一个可复现的 **pelvis/root sagittal equilibrium mismatch**。它不是视频假象，不是 X2 默认模型姿态，不是简单 default pose 错误，也不是单独 waist residual 过大。主嫌疑已收敛到迁移 actor 的腿部 sagittal target、phase 与当前高刚度稳定化控制器的联合契约。

### 下一步

1. 将 signed root pitch 按 stand/start/move/stop 写入所有 BASE 固定评估，不再只报 unsigned tilt；
2. 保留 Stage351 为相对基线，任何候选必须在 survival/行走/停止门不退化的前提下，使四阶段 signed pitch **逐阶段相对更大（更少后仰）**；
3. 下一项最高信息增益实验不是继续扫 waist/default/PD，而是单变量、低维、有界的 sagittal-chain target correction（髋 pitch + 踝 pitch 的协调修正），先验证 pelvis 巡航平衡点能否移动；
4. 在取得官方 locomotion trace 或自然站姿规范前，不设置绝对角度门，也不把 shipped dance 的 +12.63° 当目标。

训练继续锁定：signed posture 与原 BASE 生存/方向/停止门尚未共同通过。

## 6. 产物与复现

- 分析器：`tools/official_x2/analyze_sagittal_posture.py`
- 纯测试：`tests/test_sagittal_posture_audit.py`
- 完整机器可读结果：`reports/baseline/x2_recovery_phase7_sagittal_posture.json`
- 测试：6 passed（含既有 renderer pitch contract 测试）

没有修改 adapter 的 snapshot、restore 或 symmetry 代码。
