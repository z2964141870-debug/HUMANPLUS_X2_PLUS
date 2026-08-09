# X2 Forefoot AimDK v1.0 Official-Physics Screen

- 结论：**DO_NOT_PROMOTE_FROM_THIS_SCREEN**。
- 本实验不训练，不载入 checkpoint；只比较同一批 reference 在官方 MuJoCo/PD 下的物理响应。
- `prescribed_root_trackability` 的 root 每个 1 ms 被外部写回，仅回答关节/力矩可跟踪性，**不能证明平衡或可部署**。
- `free_root_balance` 不给 root 外力，仅回答开环 PD 下的初步动力学兼容性；它仍不等价于闭环策略成功。

## 假设 / 干预 / 对照

- 假设：toe→forefoot 的语义修正若有真实物理价值，应在不恶化固定-root关节可跟踪性的同时，提高或至少不降低 free-root 生存。
- 干预：只替换 `current_v4_exact30` 为 `toe_to_forefoot_exact30_smooth9` reference。
- 对照：官方 scene/model、1 kHz physics、50 Hz command、官方 29DOF RL Kp/Kd、官方 actuator torque limits 完全相同；头部按同一官方默认 20/1 锁定。
- Panel：walk straight、left/right turn、stand→walk、walk→stand。

## 聚合结果

| variant | mode | full | duration | leg RMSE(rad) | sat. | root XY RMSE(m) | slip p95(m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_v4_exact30 | prescribed_root_trackability | 5/5 | 1.000 | 0.1583 | 0.0080 | 0.0000 | 0.7764 |
| current_v4_exact30 | free_root_balance | 0/5 | 0.236 | 0.1456 | 0.0000 | 0.2675 | 0.0739 |
| toe_to_forefoot_exact30_smooth9 | prescribed_root_trackability | 5/5 | 1.000 | 0.1233 | 0.0038 | 0.0000 | 0.9752 |
| toe_to_forefoot_exact30_smooth9 | free_root_balance | 0/5 | 0.126 | 0.1240 | 0.0000 | 0.3007 | 0.1571 |

## 门禁判断

- prescribed-root trackability 不劣化：`True`。
- free-root balance 不劣化：`False`。
- 既有离线连续性门通过：`False`。
- toe-forefoot 可晋升：`False`。

即使 fixed-root 指标较好，也不得将其写成‘动作动力学可执行’；free-root 的失败也只说明开环 PD reference 不自稳，不能单独否定闭环 RL policy。

## 结果 / 结论 / 下一步

- 结果：toe→forefoot 在 prescribed-root 下更容易被官方 PD 跟踪，但在 5/5 free-root 动作中都更早失稳，且接触滑移更大。
- 结论：它改善了局部几何/关节目标，却没有改善 root-contact 闭环；不能用 fixed-root 的好看数字覆盖 free-root 退化。
- 下一步：训练数据继续保留 current-v4；不训练 smooth9。若再修 reference，应联合优化 root/COM、接触时序和足部目标，而不是继续做 foot-only offset/smoothing。

## 逐动作结果

| role | variant | mode | duration | fall(s) | leg RMSE | sat. | root XY RMSE |
|---|---|---|---:|---:|---:|---:|---:|
| walk_straight | current_v4_exact30 | prescribed_root_trackability | 1.000 | - | 0.0349 | 0.0004 | 0.0000 |
| walk_straight | current_v4_exact30 | free_root_balance | 0.195 | 1.557 | 0.1146 | 0.0000 | 0.1932 |
| walk_straight | toe_to_forefoot_exact30_smooth9 | prescribed_root_trackability | 1.000 | - | 0.0358 | 0.0001 | 0.0000 |
| walk_straight | toe_to_forefoot_exact30_smooth9 | free_root_balance | 0.098 | 0.778 | 0.0898 | 0.0000 | 0.1947 |
| turn_left | current_v4_exact30 | prescribed_root_trackability | 1.000 | - | 0.1970 | 0.0108 | 0.0000 |
| turn_left | current_v4_exact30 | free_root_balance | 0.237 | 1.890 | 0.1890 | 0.0000 | 0.5690 |
| turn_left | toe_to_forefoot_exact30_smooth9 | prescribed_root_trackability | 1.000 | - | 0.1563 | 0.0049 | 0.0000 |
| turn_left | toe_to_forefoot_exact30_smooth9 | free_root_balance | 0.115 | 0.919 | 0.1991 | 0.0000 | 0.4659 |
| turn_right | current_v4_exact30 | prescribed_root_trackability | 1.000 | - | 0.1947 | 0.0123 | 0.0000 |
| turn_right | current_v4_exact30 | free_root_balance | 0.140 | 1.117 | 0.1326 | 0.0000 | 0.1312 |
| turn_right | toe_to_forefoot_exact30_smooth9 | prescribed_root_trackability | 1.000 | - | 0.1464 | 0.0066 | 0.0000 |
| turn_right | toe_to_forefoot_exact30_smooth9 | free_root_balance | 0.099 | 0.790 | 0.1009 | 0.0000 | 0.1913 |
| stand_to_walk | current_v4_exact30 | prescribed_root_trackability | 1.000 | - | 0.1839 | 0.0055 | 0.0000 |
| stand_to_walk | current_v4_exact30 | free_root_balance | 0.260 | 1.603 | 0.1083 | 0.0000 | 0.1696 |
| stand_to_walk | toe_to_forefoot_exact30_smooth9 | prescribed_root_trackability | 1.000 | - | 0.1365 | 0.0018 | 0.0000 |
| stand_to_walk | toe_to_forefoot_exact30_smooth9 | free_root_balance | 0.118 | 0.726 | 0.0467 | 0.0000 | 0.1989 |
| walk_to_stand | current_v4_exact30 | prescribed_root_trackability | 1.000 | - | 0.1810 | 0.0109 | 0.0000 |
| walk_to_stand | current_v4_exact30 | free_root_balance | 0.347 | 1.952 | 0.1837 | 0.0000 | 0.2744 |
| walk_to_stand | toe_to_forefoot_exact30_smooth9 | prescribed_root_trackability | 1.000 | - | 0.1416 | 0.0053 | 0.0000 |
| walk_to_stand | toe_to_forefoot_exact30_smooth9 | free_root_balance | 0.198 | 1.116 | 0.1833 | 0.0000 | 0.4525 |

## 资产契约

- scene: `/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml` (`7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63`)
- x2.xml: `/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml` (`3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3`)
- motion_control.yaml: `/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml` (`07fa8a0151d14c2b0b9f8659302f96dec21fe3b3921f9dbcaab1150dc1c2e400`)
- physics/control period: `0.001` / `0.02` s
- head contract: `motion_control.yaml default_kp/default_kd=20/1`
