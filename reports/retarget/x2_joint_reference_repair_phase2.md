# X2 Joint Reference Repair Phase2

- 裁决：**JOINT_REPAIR_ORACLE_NOT_SUPPORTED**。
- 本轮不训练、不载入 checkpoint、不接真机；唯一主干预是低维联合 reference repair。
- COM、DCM、contact probability、支撑相位全部来自官方 AimDK v1.0 MuJoCo 模型估计，**不是真实足底力、COP 或实机 COM 真值**。
- oracle 使用完整动作和官方物理结果选参，只用于回答‘是否存在可行修复’，不是可部署在线 adapter。

## 假设 / 干预 / 对照

- 假设：foot-only 修正失败，是因为 root/COM、接触时序、支撑脚锁定和摆脚 terminal 没有被联合处理。
- 干预：在 current-v4 exact30 上联合搜索有界 root 支撑偏移/高度、±2 帧相位、stance-foot IK lock、0–2 cm 摆脚 clearance 与 0–25% foot-terminal blend。
- 对照：current-v4 exact30 与已淘汰 toe-smooth9；官方 scene、PD、限矩、1 kHz physics、50 Hz control 完全相同。

## 离线门

| role | candidates | passed | selected offline cost | root XY/Z max(m) | joint step(rad) |
|---|---:|---:|---:|---:|---:|
| walk_straight | 12 | 12 | 3.1684 | 0.0100/0.0000 | 0.0893 |
| turn_left | 12 | 10 | 3.3887 | 0.0100/0.0158 | 0.1614 |
| turn_right | 12 | 9 | 3.4885 | 0.0100/0.0169 | 0.1554 |
| stand_to_walk | 12 | 9 | 3.7870 | 0.0100/0.0000 | 0.1466 |
| walk_to_stand | 12 | 10 | 3.8913 | 0.0100/0.0000 | 0.1401 |

## 官方物理结果

| variant | mode | full | duration | leg RMSE | sat. | slip p95 | root XY RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| current_v4_exact30 | prescribed_root_trackability | 5/5 | 1.000 | 0.1583 | 0.0080 | 0.7764 | 0.0000 |
| current_v4_exact30 | free_root_balance | 0/5 | 0.236 | 0.1456 | 0.0000 | 0.0739 | 0.2675 |
| toe_to_forefoot_exact30_smooth9 | prescribed_root_trackability | 5/5 | 1.000 | 0.1233 | 0.0038 | 0.9752 | 0.0000 |
| toe_to_forefoot_exact30_smooth9 | free_root_balance | 0/5 | 0.126 | 0.1240 | 0.0000 | 0.1571 | 0.3007 |
| joint_repair_oracle | prescribed_root_trackability | 5/5 | 1.000 | 0.1406 | 0.0090 | 1.3316 | 0.0000 |
| joint_repair_oracle | free_root_balance | 0/5 | 0.199 | 0.1362 | 0.0000 | 0.0671 | 0.2359 |

## 逐动作 free-root 对照

| role | variant | duration | fall(s) | slip p95 mean(m/s) | root XY RMSE(m) |
|---|---|---:|---:|---:|---:|
| walk_straight | current_v4_exact30 | 0.195 | 1.557 | 0.0099 | 0.1932 |
| walk_straight | toe_to_forefoot_exact30_smooth9 | 0.098 | 0.778 | 0.0187 | 0.1947 |
| walk_straight | joint_repair_oracle | 0.157 | 1.253 | 0.0107 | 0.2065 |
| turn_left | current_v4_exact30 | 0.237 | 1.890 | 0.1878 | 0.5690 |
| turn_left | toe_to_forefoot_exact30_smooth9 | 0.115 | 0.919 | 0.1839 | 0.4659 |
| turn_left | joint_repair_oracle | 0.178 | 1.422 | 0.1665 | 0.3195 |
| turn_right | current_v4_exact30 | 0.140 | 1.117 | 0.0964 | 0.1312 |
| turn_right | toe_to_forefoot_exact30_smooth9 | 0.099 | 0.790 | 0.1575 | 0.1913 |
| turn_right | joint_repair_oracle | 0.124 | 0.986 | 0.0695 | 0.1662 |
| stand_to_walk | current_v4_exact30 | 0.260 | 1.603 | 0.0059 | 0.1696 |
| stand_to_walk | toe_to_forefoot_exact30_smooth9 | 0.118 | 0.726 | 0.1629 | 0.1989 |
| stand_to_walk | joint_repair_oracle | 0.182 | 1.124 | 0.0097 | 0.2128 |
| walk_to_stand | current_v4_exact30 | 0.347 | 1.952 | 0.0695 | 0.2744 |
| walk_to_stand | toe_to_forefoot_exact30_smooth9 | 0.198 | 1.116 | 0.2628 | 0.4525 |
| walk_to_stand | joint_repair_oracle | 0.351 | 1.976 | 0.0788 | 0.2746 |

## 结果 / 结论 / 下一步

- 结果：联合 oracle 未同时满足 free-root 生存、滑移与 fixed-root 跟踪三项门禁。
- 结论：现有低维参数化不足以证明 reference 可由短窗口联合修复变成动力学可行数据；不得解锁 WBT PPO。
- 下一步：停止脚部/平滑参数扫描，转向显式接触约束的轨迹优化或目标机器人原生动态 teacher。

## 严格解释边界

`prescribed_root_trackability` 每个物理步回写 root，只证明关节/力矩可跟踪；`free_root_balance` 不施加 root 外力，但仍只是开环 PD reference 可行性，不等价于闭环 RL 成功。
