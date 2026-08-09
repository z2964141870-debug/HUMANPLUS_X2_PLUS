# X2 Physics Provenance / B5 — Phase25

- 裁决：**B5_READY_AS_DECLARED_TRAIN_DOMAIN**。
- IsaacLab被声明为训练域；AimDK v1 MuJoCo只作为held-out sim-to-sim mismatch域，绝不宣称二者物理等价。
- 本阶段只有静态loader/FK/default/limit/PD与hash guard；没有physics stepping、训练或zero-update。

## 假设

训练域不必数值复制官方MuJoCo，但必须冻结asset生成链、控制与solver合同，并把全部已知差异显式暴露。

## 干预与对照

- 干预：冻结raw URDF→sole12 builder→sole12 URDF、mesh树、Isaac converter/spawner、x2/env代码和完整runtime snapshot。
- 对照：官方AimDK v1 `x2.xml`、`motion_control.yaml`、scene/simulator配置和同一mesh树。

## 逐项结果

| 项目 | 状态 | 证据摘要 |
|---|---|---|
| mesh tree | matched | `{"45_files_same_hash":true}` |
| 31 joint names | matched | `{"same_set":true}` |
| neutral kinematic FK | matched | `{"pos_max_m":5.452424987951902e-07,"ori_max_rad":8.873388130044874e-07}` |
| non-pelvis mass/principal inertia | matched | `{"bodies":31,"mass_max":3.799999999998249e-05,"inertia_max":4.6410146498088167e-07}` |
| pelvis mass | different | `{"isaac_urdf":3.523487,"official_compiled":5.031810659272607}` |
| joint position limits | matched | `{"max_abs_rad":0.0}` |
| effort limits | matched | `{"max_abs_nm":0.0}` |
| velocity limits | unknown | `"official x2.xml/motion_control does not declare velocity limits"` |
| active sole geometry | matched | `{"12_spheres_each":true}` |
| full active collision set | different | `{"isaac":52,"official":49,"body_count_differences":{"head_yaw_link":{"isaac_source":1,"official":0},"left_wrist_yaw_link":{"isaac_source":1,"official":0},"right_wrist_yaw_link":...` |
| default pose/root height | different | `{"joint_max_rad":0.8999999999999999,"isaac_root_z":0.65,"official_xml_root_z":0.68}` |
| application PD | different | `{"kp_max":185.74937690131108,"kd_max":2.4421102348989447}` |
| joint armature | different | `{"max_abs":0.026390274999999998}` |
| action scale | different | `{"max_abs_29":0.592068442684729}` |
| control dt | matched | `{"isaac":0.02,"official":0.02}` |
| physics dt | different | `{"isaac":0.005,"official_mjcf":0.001}` |
| cross-engine solver | unknown | `"PhysX iteration counts and MuJoCo solver are not numerically equivalent parameters"` |
| raw URDF official generation lineage | unknown | `"mesh and most inertials match, but no signed generator/provenance states x2.xml was generated from this URDF"` |

## 关键边界

- pelvis mass：URDF `3.523487kg` vs official compiled `5.031811kg`。
- active collision：sole12训练域 `52` vs official `49`；足底12球逐项匹配 `True`。
- neutral FK max：`5.45e-07m/8.87e-07rad`；各自default pose FK差异不是loader错误，max `0.130m`。
- PD/action-scale/physics-dt差异保留为held-out mismatch，不进入`matched`。

## Guard

- 文件/mesh hash：`{'runtime_snapshot': True, 'runtime_guard': True, 'raw_x2_urdf': True, 'training_sole12_urdf': True, 'sole12_builder': True, 'local_mesh_tree': True, 'x2_robot_config': True, 'modular_env_config': True, 'isaac_urdf_converter': True, 'isaac_urdf_spawner': True, 'official_x2_xml': True, 'official_motion_control': True, 'official_scene': True, 'official_simulator_config': True, 'official_mesh_tree': True}`。
- runtime exact：`True`；负例sim_dt改写被拒：`True`。
- dedicated launcher不读物理环境变量；启动前必须通过manifest hash+resolved runtime snapshot guard。

## 结论

- 结果：The sole12 Isaac training domain is immutable and prelaunch hash/runtime guarded; all official mismatches are explicit.
- 结论：B5 is ready as a declared training domain, not as a claim of official-MuJoCo equivalence.
- 下一步：Update the faithful launcher to require this manifest hash, then stop for review before any zero-update.
