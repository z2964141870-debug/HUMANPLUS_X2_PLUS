# BASE Phase37：显式三事件摆脚 teacher 裁决

## 一句话裁决

在完整 zero-candidate matched-control 门通过后，固定三事件 teacher 只产生 26 ms 离地和 0.63 mm 净空；虽然支撑滑移、root/tilt、重新接触与后续安全都通过，但远低于 100 ms / 12 mm 硬门，因此三事件 teacher 路线停止，训练继续锁定。

## 游戏任务

- [x] 同一 Phase36 fixed mismatch window、exact snapshot 和 Phase34 future ctrl。
- [x] 100 ms anticipatory support-load shift。
- [x] 独立 swing lift 与 touchdown blend。
- [x] 360 ms teacher＋80 ms recorded-control follow-up。
- [x] 9 个有界变量，固定 seed 3601 / 24×5 / elite 6。
- [x] 完整 440 ms zero-candidate matched no-op。
- [x] 一次有效 CEM 结果，无阈值、窗口或预算扫描。
- [ ] 100 ms swing-off。
- [ ] 12 mm clearance。
- [ ] 合格 teacher / 训练解锁。

## 假设

Phase36 的局部五模式没有显式提前移重心与 touchdown 事件。若缺口主要是事件组织，那么在同一 exact 初态上加入：

1. `0–100 ms` 支撑 hip-roll＋waist-roll 预期姿态调整；
2. `100–260 ms` 摆动 hip/knee/ankle 独立 lift；
3. `260–360 ms` knee/ankle touchdown blend；

应当在保持支撑与 root 安全的同时产生 ≥100 ms 离地、≥12 mm 净空并重新落地。

## 干预与对照

窗口仍为 move 0.08 s 的右脚 mismatch，anchor t=2.310 s。事件 liftoff/touchdown 只允许各 ±20 ms；关节幅度固定有界。所有候选以 recorded ctrl 原样为基底，只叠加位置等效增量，**不再重新 clip recorded ctrl**。

preflight：

- exact recorded ctrl 前 10 ms：qpos/qvel error=0；
- zero candidate 完整 440 ms：qpos max error=`8.04e-15`、qvel=`2.04e-13`；
- 因此本轮收益不是 train/eval mismatch 或初态误差。

第一次正式调用没有 result artifact、没有可判 evaluation output，记为 `PRE_RESULT_RUNTIME_INVALID`。经授权只改变 conda 输出可观测方式，runner/hash/seed/120 eval/窗口/门槛完全不变；第二次为唯一有效结果，此后不再重试。

## 结果

| 指标 | zero | best | 硬门 | 裁决 |
|---|---:|---:|---:|---:|
| 连续 swing-off | 0 ms | 26 ms | ≥100 ms | fail |
| clearance | 0 | 0.63 mm | ≥12 mm | fail |
| stance core slip p95 | 0.0406 | 0.0281 m/s | ≤0.10 | pass |
| root-z min | 0.663 | 0.631 m | ≥0.55 | pass |
| root tilt max | 0.197 | 0.262 rad | ≤0.30 | pass |
| terminal recontact 40 ms | fail | pass | required | pass |
| follow-up 80 ms safe | pass | pass | required | pass |

120 个候选没有任何 strict success。best cost 由 16.0 降至 8.303，但主要获得了重新接触和安全，并未形成可用摆脚。

## 结论

显式三事件组织本身不足以解决这个窗口。与 Phase36 的 92 ms 数字不能直接比较，因为 Phase37 preflight 证明 Phase36 candidate 曾重新 clip recorded ctrl；Phase36 现仅保留为 diagnostic near-miss。本轮 Phase37 是第一个完整 440 ms zero-noop matched 的 teacher 搜索，其否定证据更可靠。

按预注册规则：停止该 teacher 路线，不增加 CEM 预算、不换窗口、不放松硬门，也不把 26 ms 片段当训练 label。

## 下一步

BASE 训练继续锁定。若项目仍继续下肢接触修复，应离开“recorded ctrl 上叠加几个局部关节模式”的路线，重新论证能直接约束足端/接触或优化整段目标轨迹的动力学方法；本阶段不自动启动。

## 证据边界

结果来自 exact-state direct vendor MuJoCo 3.3.7；不是 closed ROS 完整闭环，也不是实机足底力、GRF 或 COP 真值。

证据：[预注册](../official_x2/phase37_three_event_teacher_prereg.json) · [结果 JSON](../official_x2/phase37_three_event_teacher_result.json)
