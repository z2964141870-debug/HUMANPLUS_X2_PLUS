# X2 WBT Phase35：free-root exact-joint kinematic oracle

## 裁决

- **PHASE35_REFERENCE_GEOMETRY_CONTACT_PRIMARY**
- exact-joint oracle：0.324/5.800s；Phase33 free-PD：0.575/5.800s。
- oracle attribution：`REFERENCE_GEOMETRY_CONTACT_PRIMARY`。

## 假设 / 干预 / 对照

- 假设：若精确施加WBT29 q/dq后仍保持flight/快速倒，reference geometry/contact是主因；若稳定且接触显著改善，裸PD/闭环是主因。
- 干预：同一冻结Phase33 50Hz candidate与root初态；每20ms精确写WBT29 q/dq，head=model nominal，root/contact/gravity/solver自由；actuator torque恒零。
- 对照：只读取Phase33既有free-PD结果，不重跑。
- direct q/dq overwrite会破坏动量连续，故这是kinematic feasibility oracle，不是可执行controller或Gold晋升。

## 结果

- terminal：`{'time_s': 0.324, 'trigger': 'tilt', 'root_z_m': 0.5409191181184152, 'root_tilt_rad': 0.9014770596066547, 'control_frame': 16, 'substep': 4, 'q_error_rmse_max_rad': [0.002515422637198117, 0.007602945430760455], 'official_sole_collision_lr': [True, False]}`。
- q error RMSE/p95/max：`[0.01629433932415193, 0.0372587084987055, 0.12427038546874501]` rad；overwrite前jump p95/max：`[0.07130777109985248, 0.12365344163253285]` rad。
- root z min：0.5292m；tilt p95/max：`[0.844460651632292, 0.9014770596066547]` rad。
- reference contact L/R：`[0.11764705882352941, 0.0]`；realized：`[1.0, 0.8235294117647058]`；agreement L/R/mean：`[0.11764705882352941, 0.17647058823529413, 0.14705882352941177]`。
- realized DS/SS/flight：`[0.8235294117647058, 0.17647058823529413, 0.0]`。
- contact linear impulse L/R：`[155.3498797280873, 162.42484040359508]` Ns；torque impulse：`[0.0, 0.0]` Nms。
- actuator generalized force max：0；constraint p95/max：`[50.89769820670236, 388.77524244567854]`。

## 结论

即使消除29DOF关节跟踪误差，free root仍快速失败；当前主因收敛到reference geometry/contact feasibility，而不是裸PD跟踪误差。

## 下一步

停止oracle，不调PD/加warmup；回到contact-feasible reference生成合同。
