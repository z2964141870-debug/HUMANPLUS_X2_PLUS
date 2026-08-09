# X2 Observable Ready Handshake Phase19

- 裁决：**PHASE19_ZERO_MODE_TELEMETRY_ABSENT_NO_CAPTURE**。
- 唯一domain218运行只观察ready；未达到ready后严格没有发布JOINT/RL、没有生成NPZ、没有运行physics。

## 假设

若Phase17只是5秒窗口太短，则在不改变snapshot_ready条件的15秒窗口中，应观察到关节/odom/IMU逐步到达并最终ready。

## 干预 / 对照

- 干预：只增加readiness diagnostics，不改变订阅、数据缓存、控制或mode路径。
- 对照：Phase17没有缺项可观测性，只知道5秒内ready未产生。
- 诊断字段：已见joint names/count、缺失31列表、各group计数、odom/IMU seen与首达时间；15秒仍由真实snapshot_ready硬触发。

## 结果

- 观察时长：`15.164s`；snapshot_ready：`False`。
- state：`0/31`；leg/waist/arm/head分别为 `0/0/0/0`。
- odom seen：`False`；IMU seen：`False`；所有首达时间均为空。
- controller模型加载和sim初始化日志存在；没有自发发布JOINT/RL，符合严格握手设计。

## Gate

- `diagnostic_reached_at_least_15s`: `True`
- `snapshot_ready`: `False`
- `state_31_complete`: `False`
- `odom_seen`: `False`
- `imu_seen`: `False`
- `all_state_groups_zero`: `True`
- `controller_loaded`: `True`
- `simulator_initialized`: `True`
- `joint_mode_not_published`: `True`
- `rl_mode_not_published`: `True`
- `npz_not_written`: `True`
- `manifest_not_written`: `True`
- `timeout_diagnostic_present_in_log`: `True`

- ready qualification：`False`；prescribed/free executed：`False/False`。

## 结论

- 结果：After 15.090 s, state remained 0/31 across every group, odom and IMU were unseen, and snapshot_ready remained false.
- 结论：In the current official stack and 15 s bound, telemetry required for a complete reference does not start before JOINT_DEFAULT; a complete-ready-before-JOINT handshake is therefore not implementable as specified.
- 下一步：Stop without physics. If separately authorized, publish JOINT_DEFAULT to activate telemetry, wait for the first real complete snapshot, then hold JOINT for four additional scoreable seconds before RL; do not use topic existence or sleep as ready.
