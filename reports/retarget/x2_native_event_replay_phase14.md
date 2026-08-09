# Official X2 Event-Replay Recorder Phase14

- 裁决：**PHASE14_WBT29_EVENT_CAPTURE_PASSED_WITH_RECORDER_SOURCE_DRIFT_BOUNDARY**。
- 范围：一次隔离的 AimDK v1.0 官方 MuJoCo + 官方 ONNX 20s 采集的离线审计；未训练、未做物理 replay、未上真机。
- 边界：这是官方仿真 truth，不是真机 GRF/COP；source rollout 自身稳定不能冒充 replay 稳定。

## 假设

若逐 callback receipt 事件、各组局部索引、full31 latest-map/valid mask、50Hz 状态快照及模型/配置/scene/recorder 身份都可验证，则 Phase13 的 active-WBT29 事件时序缺口可关闭。

## 干预 / 对照

- 对照：Phase13 的 50Hz latest-value 快照，缺少完整 500Hz command event stream。
- 干预：独立 v2 recorder；逐 leg/waist/arm/head callback 记录 monotonic receipt、global/group index、header字段、changed mask、full31 latest-map；另存50Hz state/odom/IMU callback receipt。
- 本阶段只审计采集完整性；没有拿 command 当 reference，也没有做 replay 性能结论。

## 结果

- snapshot：`999` 帧，`19.960s`，`50.000Hz`；dt median/p95/max = `0.019982/0.020808/0.021721s`。
- command events：`40083`，aggregate `2004.20Hz`（四组合计）。
- `leg`：`10022` events，`501.074Hz`，interval p50/p95/p99/max = `1.960/2.645/4.667/12.302ms`，changed joints `{'12': 10022}`。
- `waist`：`10021` events，`501.064Hz`，interval p50/p95/p99/max = `1.953/2.691/4.761/12.398ms`，changed joints `{'3': 10021}`。
- `arm`：`10020` events，`501.016Hz`，interval p50/p95/p99/max = `1.949/2.728/4.737/12.499ms`，changed joints `{'14': 10020}`。
- `head`：`10020` events，`501.015Hz`，interval p50/p95/p99/max = `1.950/2.672/4.801/12.557ms`，changed joints `{'0': 10020}`。

## Group ordering 边界

- global index 与 receipt-monotonic 顺序完整，per-group index 均从0连续；后续 replay 应以保存的 global receipt order 为准。
- 开头40条呈 `leg→waist→arm→head` 周期，但线程调度会改变相邻到达顺序；最常见 transitions：`[{'order': ['leg', 'waist'], 'count': 9583}, {'order': ['arm', 'head'], 'count': 9305}, {'order': ['head', 'leg'], 'count': 9279}, {'order': ['waist', 'arm'], 'count': 9092}, {'order': ['waist', 'head'], 'count': 714}]`。
- 因此四组不是可证明的 publisher-atomic cycle，不能把固定 leg/waist/arm/head 顺序硬编码为真值。

## Header 与 31/29DOF 边界

- message schema 中 header 对象存在率 `1.000`，但 populated率 `0.000`；stamp/sequence 全零：`True`。
- 唯一权威时间是 subscriber callback 的 `time.monotonic_ns()`；无法恢复 publisher/source timestamp 或 publisher sequence。
- 数组宽度为31，但事件 warmup 后与全部snapshot都只有29个valid关节；`head` 有 `10020` 条空callback，changed joints恒为0。
- 所以该资产可描述 official WBT29 active-control contract；不能声称获得了31个主动关节或头部控制轨迹。

## 身份与 Gate

- ROS domain `214`，control mode `RL_DEFAULT`；NPZ、ONNX、control YAML、scene 的捕获证据可验证。
- recorder capture SHA `d7a2888ff4141180282e139062720528f2318d4c650de8d17b41e3c60df8bfac`；current SHA `fa6671c38a0d4fed843c64d19e55f5a2320b66a6473b018f132bc46ee3b09087`；匹配：`False`。
- 捕获时 recorder 源码未另存快照；因此当前漂移不等于 capture 数据损坏，但不能声称 recorder 源码可精确重现。
- Phase15 只依赖 manifest 绑定的 NPZ 事件数组，不依赖 current recorder SHA；其既有裁决不受这次源码漂移影响。
- `npz_hash_matches_manifest`: `True`
- `capture_runtime_artifacts_recoverable_exactly`: `True`
- `current_recorder_source_matches_capture_manifest`: `False`
- `schema_keys_complete`: `True`
- `manifest_counts_match`: `True`
- `global_event_index_exact`: `True`
- `global_receipt_monotonic`: `True`
- `elapsed_matches_receipt_origin`: `True`
- `all_group_local_indices_contiguous`: `True`
- `all_group_rates_approximately_500hz`: `True`
- `changed_joint_contract_exact`: `True`
- `snapshot_rate_approximately_50hz`: `True`
- `snapshot_receipt_monotonic`: `True`
- `legacy_arrays_finite`: `True`
- `root_quaternion_normalized`: `True`
- `controller_log_proves_joint_then_rl_mode`: `True`
- `header_schema_present`: `True`
- `publisher_header_populated`: `False`
- `effective_wbt29_after_warmup`: `True`
- `active_31_joint_control_available`: `False`

## 结论

- 结果：The immutable capture data passes event/schema/timing integrity and its ONNX/config/scene bytes remain recoverable. The current recorder source has drifted from the capture-time digest and the historical source was not archived separately.
- 结论：Phase13 event timing/provenance gap is closed for active WBT29 subscriber-receipt replay, but not for publisher-time exactness or active head control.
- 下一步：Stop at Phase14. Do not run Phase15 until explicitly authorized; any later replay must preserve receipt order and validity masks and must not claim source-trace stability as replay stability.
