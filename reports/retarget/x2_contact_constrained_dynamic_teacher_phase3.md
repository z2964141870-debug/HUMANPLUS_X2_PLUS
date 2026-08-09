# X2 Contact-Constrained Dynamic Teacher Phase3

- 裁决：**PANEL_DYNAMIC_TEACHER_NOT_SUPPORTED**。
- walk smoke 先独立通过后才扩展四动作；全程无训练、无 checkpoint、无真机。
- COM、DCM、contact、支撑相位均为官方 AimDK v1.0 MuJoCo 模型估计，并非真实 GRF/COP/COM/contact。
- 所有 teacher 都使用完整未来与 free-root 物理选参，是 existence oracle，不可部署。

## 可证伪契约

优化变量覆盖 COM 横向/root 高度样条、接触切换、stance terminal 零速约束、swing clearance 与 landing terminal；free-root 代价覆盖生存、root tilt/DCM、slip、接触一致、action/torque 平滑。单动作晋升要求 `+0.5 s` 生存、slip 不超过容差且 prescribed-root 平滑/未饱和。

## 五动作结果

| role | generated/pass/rollout | baseline(s) | teacher(s) | gain(s) | baseline/teacher slip | action Δ | sat. | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walk_straight | 72/41/41 | 1.557 | 2.456 | 0.899 | 0.0087/0.0066 | 0.0617 | 0.0018 | True |
| turn_left | 12/0/0 | 1.890 | n/a | n/a | 0.1293/n/a | n/a | n/a | False |
| turn_right | 36/20/20 | 1.117 | 1.197 | 0.080 | 0.1007/0.0248 | 0.1217 | 0.0116 | False |
| stand_to_walk | 36/34/34 | 1.603 | 1.564 | -0.039 | 0.0033/0.0064 | 0.1221 | 0.0074 | False |
| walk_to_stand | 36/17/17 | 1.952 | 2.101 | 0.149 | 0.0658/0.1066 | 0.1712 | 0.0123 | False |

## 方法学自查

- walk teacher 证据：single clip, one CEM seed, deterministic official MuJoCo; no independent held-out clip, no multi-seed physics and still falls after the optimized short window。
- 初始状态契约：qpos is exact reference frame 0 but qvel is forced to zero; nonzero frame-0 reference joint/root velocity is therefore not restored and can bias early failure。
- 接触契约：expected contact is inferred from model FK height/speed, not ground truth; 0.68-0.85 match on turn/start-stop actions indicates a likely schedule/official-contact mismatch。
- 表达能力限制：the teacher optimizes COM lateral/root height and foot XYZ, but not root yaw, foot orientation, or centroidal angular momentum; turn_left failing terminal IK is not a clean dynamics impossibility proof。

| role | first swing(s) | baseline fall(s) | fall before event | contact match | qdot0 p95/max(rad/s) |
|---|---:|---:|---:|---:|---:|
| walk_straight | 2.200 | 1.557 | True | 0.965 | 2.239/2.680 |
| turn_left | 0.633 | 1.890 | False | 0.680 | 0.783/0.991 |
| turn_right | 0.300 | 1.117 | False | 0.719 | 0.900/1.193 |
| stand_to_walk | 1.167 | 1.603 | False | 0.852 | 0.347/0.385 |
| walk_to_stand | 0.533 | 1.952 | False | 0.710 | 0.922/1.208 |

## 结果 / 结论 / 下一步

- 结果：1/5 动作同时通过生存、滑移和 prescribed-root 门。
- 结论：walk 的单样本正结果未达到跨动作、held-out 或多seed稳健性，不能生成训练 Silver 集或解锁 PPO。
- 下一步：停止扩大搜索；先修正初始qvel与接触事件/转向姿态契约，再重新预注册验证。
