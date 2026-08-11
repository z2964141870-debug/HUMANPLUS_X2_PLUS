# X2 DSMS Phase 3b：合同修正与唯一短前缀求解

日期：2026-08-11
状态：**CORRECTED_CONTRACT_PASSED / NLP_AND_RAW_REPLAY_REJECTED**
训练：0；PPO/Phase 4：未运行；参数扫描：0。

## 结论先行

上一轮 Phase 3 的“DSMS 已被否证”不成立，因为实现同时存在参考速度全零、官方 PD 错配、head 约束未注册、head bounds 过期、官方 Euler 被 `implicitfast` 替换等合同错误。Phase 3 应标为 `INVALID_CONTRACT_IMPLEMENTATION`。

本轮修复全部合同后，唯一一次 0.34 s 短前缀求解仍未得到可行证书：IPOPT/MUMPS 在 300 轮后停止，返回 best iterate 的最大 shooting defect 为 `0.05422`（门 `0.01`）。raw official motor 独立重放又同时失败于穿透、支撑脚速度、flight 和 root 加速度。因此，当前可诚实否定的是：

> “上游 `g1_gait` 原始 tracking cost + X2 官方 PD-servo 等价表示 + 18 帧短前缀”不能把 Phase30 lunge 修成合格的 X2 动态参考。

这不是对 DSMS 论文方法、X2 本体或动态重定向可行性的总体否证。

## 1. 修正内容

- Phase30 30 Hz reference 按 Phase33 合同重采到 50 Hz；关节、root 线速度和 quaternion-aware 角速度均非零。
- 优化器内部把官方 torque motor 精确改写为逐子步 PD position-servo；raw model 仍保持 motor，用于独立重放。
- PD 来自官方 `motion_control.yaml`，body29 逐关节读取，head2 使用官方默认 `20/1` 并固定 target=0。
- 保留官方 MuJoCo Euler、1 kHz physics、50 Hz control。
- 约束返回/注册长度严格一致；head 控制边界在 IPOPT bounds 构建前固定。
- 未修改 upstream `g1_gait` cost 权重；没有调 solver、阈值或前缀长度。

## 2. Preflight

| 门 | 结果 |
|---|---:|
| MuJoCo | 3.3.7 |
| 官方 Euler | PASS |
| reference velocity max | 2.458756 rad/s |
| 官方 PD exact | PASS |
| constraints | 1350 returned / 1350 registered |
| head target bounds | `[0,0]`, `[0,0]` |
| raw external-PD vs servo，100 ms qpos | 1.11e-16 |
| raw external-PD vs servo，100 ms qvel | 1.20e-14 |
| Phase33 free-root baseline | 0.575 s（精确复现） |

## 3. 唯一 NLP 结果

- 变量 1872，等式约束 1350；IPOPT 3.14.19 + MUMPS 5.8.2。
- 300 iterations / 443.31 s；status `-1`，maximum iterations exceeded。
- 最后一轮：objective `0.05885`，constraint violation `0.24472`。
- keep-best 返回第 262 轮：max defect `0.0542203`。
- defect p95 `0.003414`，但最大值由 `left_wrist_pitch` velocity defect 主导；分块最大值：root `0.00846`、lower+waist `0.02686`、arms `0.05422`、head `6.89e-05`。
- 因采用“最大 defect <=0.01”硬门，不能用 p95 掩盖局部不连续，NLP gate 失败。

## 4. raw official motor 独立重放

候选控制在未改写的官方 motor 模型中以逐子步外部 PD 扭矩执行：

| 指标 | 结果 | 门 |
|---|---:|---:|
| 0.34 s 前缀存活 | PASS | full |
| root-z min | 0.5467 m | 诊断 |
| tilt max | 0.4480 rad | 诊断 |
| torque saturation | 1.689% | 诊断 |
| terminal qpos vs stitched | 0.001261 rad | 诊断 |
| active-sole penetration min L/R | -3.54 / -5.03 mm | >= -0.5 mm |
| stance speed p95 L/R | 0.1300 / 0.1906 m/s | <=0.10 |
| stance excursion max L/R | 0.02492 / 0.01842 m | <=0.03 |
| contact contradiction L/R | 38.44% / 31.76% | 诊断 |
| unintended flight | 25.29% | <=2% |
| root horizontal accel p95/max | 11.71 / 21.32 m/s^2 | p95<=4 |
| head max | 0.000355 rad | <=0.02 |

只有短前缀存活、stance excursion 和 head 三类门通过；总体 `false`。首次报告曾把每脚一个 `contype=0` visual mesh 混入 sole geometry，产生 `-25.80/-48.66 mm` 假值；此处与 JSON 已按官方 12 个 active sole spheres 更正，其他指标和拒绝裁决不变。

## 5. 根因边界

上游 `G1GaitTO` 的硬约束只有初态、shooting dynamics defects 和 control box；objective 是 state/EE tracking、torque 与 control-rate。它没有显式约束“source intent 支撑脚必须进入官方 sole contact 且保持低速”。与此同时，Phase30 reference 的官方 collision 语义本身接近全程 flight。因而低 tracking cost 不等于 contact-consistent reference，这与 raw replay 的穿透/flight/高足速结果一致。

这支持下一阶段重新预注册“contact-aware objective/constraint 或换动态重定向表示”，不支持继续增加同一 cost 的迭代数，也不支持进入 RL。

## 6. 停止与保留产物

- 按预注册停止：不追加 solver、cost、prefix 或阈值扫描。
- Phase 4 RL 保持禁止。
- 保留 preflight、唯一 solve、candidate NPZ、raw replay JSON 和本报告，供下一种动态重定向方法作对照。

主要文件：

- `phase3b/prereg_phase3b_corrected.json`
- `phase3b/phase3b_preflight.json`
- `phase3b/phase3b_solve_result.json`
- `phase3b/x2_lunge_phase3b_prefix.npz`
- `phase3b/phase3b_raw_motor_replay.json`
- `phase3b/x2_lunge_dsms_phase3b_corrected.py`
- `phase3b/replay_x2_lunge_phase3b_raw.py`
