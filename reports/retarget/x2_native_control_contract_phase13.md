# X2 Native Source Control Contract Phase13

- 裁决：**PHASE13_CONTROL_CONTRACT_INCOMPLETE_STOP**。
- 只审计 Phase10 源 NPZ 与采集/官方发布代码；因控制事件时序合同不完整，未执行 prescribed/free 物理 A/B。
- command 仍只作控制输入候选，评分 reference 始终应是 recorded actual q/root/contact；本阶段没有把 command 冒充 reference。

## 假设

若源 NPZ 保存了 source policy 真正发布的逐控制事件 command、Kp/Kd、关节顺序、原始消息时间戳/sequence 和可验证的模型配置身份，才允许把 Phase12 的 synthetic PD 换成 recorded control contract 做单变量 A/B。

## 干预 / 对照

- 对照：Phase10 源 NPZ/manifest 与原始采集器实现。
- 干预：无；只读审计 shapes、finite、ordering、sample timing、ROS topic provenance、publisher/recorder语义。
- 停止条件：缺少原始 command event timestamp/sequence、组间原子同步或 source policy/config identity 任一项，即不进入物理。

## 已保存内容

- `3000` 帧 × `31` 关节；snapshot time dt mean/std/min/max = `0.020000/0.000590/0.015696/0.023904s`。
- command valid fraction `1.000`；Kp unique `[0.0, 20.0, 40.0, 40.17919921875, 50.0, 100.0, 120.0, 150.0, 200.0]`；Kd unique `[0.0, 2.0, 2.5578999519348145, 3.0, 4.0, 5.0]`。
- archive joint order exact recorder contract：`True`；command q/dq/effort/Kp/Kd/valid 形状与有限性：`True`。
- official topic provenance：`True`；官方 C++ 会把 position/stiffness/damping 写入这些 topic：`True`。

## 缺失的可重放合同

- 官方 publisher 500Hz；采集器只用独立 50Hz timer 读取四个 topic 的 latest-value 字典，因此 NPZ 是异步快照，不是完整 500Hz command event stream。
- `JointCommandArray` 定义含 stamp/sequence，但官方 `createCommand()` 没有填 header；采集器也未保存 header、callback receipt time 或每组 sequence。
- arm/leg/waist/head 四组独立 callback 更新同一字典，NPZ 未保存组级到达时刻，无法证明同一行31关节来自同一 publish cycle。
- manifest 有人类可读 source 标签，但没有逐帧 control mode、ONNX hash、control YAML hash、process/launch identity；不能仅凭标签完整证明每一帧来自指定 policy/config。

## Gate

- `source_manifest_hash_matches`: `True`
- `all_command_arrays_complete`: `True`
- `all_joint_commands_valid`: `True`
- `archive_joint_order_exact_recorder`: `True`
- `snapshot_timestamps_monotonic`: `True`
- `values_are_sampled_from_official_command_topics`: `True`
- `official_publisher_emits_control_fields`: `True`
- `original_command_event_timestamp_preserved`: `False`
- `original_command_sequence_preserved`: `False`
- `four_group_atomic_cycle_preserved`: `False`
- `source_policy_and_config_identity_hashes_preserved`: `False`

## 结论

- 结果：q/dq/effort/Kp/Kd 值与关节顺序完整且来自官方 command topics，但缺少原始500Hz事件时间戳/sequence、四组原子同步和policy/config身份hash；按规则未运行物理。
- 结论：现有NPZ适合command-to-state响应证据，不能被提升为可精确复现source闭环的控制事件流；这不否定Gold、Any2Any或官方policy。
- 下一步：若未来可重录，保存四个topic每条消息的callback monotonic time、header/sequence、group、mode、ONNX/config hash；在此之前不做Phase13 physics。
