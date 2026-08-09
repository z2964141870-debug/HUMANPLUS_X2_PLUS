# BASE Phase31：Stage250 prepare 可重建性与 closed ROS 时序审计

日期：2026-08-09
裁决：`BLOCKED_BY_UNRECORDED_PREPARE_AND_APPLICATION_TIMING`
边界：只读；0 个新 physics probe，0 训练，不碰 WBT/Git/百度/真机。

## 游戏任务卡

- [x] 审计 Stage250 首个完整 stand row 前 0.2 秒 prepare 的逐行字段。
- [x] 审计 launcher 与 adapter 当前 prepare 公式、PD publish contract。
- [x] 判断真实 prepare command/q/q̇/contact/ctrl 是否可唯一重建。
- [x] 因不可重建，禁止猜测式 direct prepare probe。
- [x] 转而量化三份 existing event-v2 的 command receipt 与 telemetry sampling offset。
- [x] 裁决 event-v2 能否支持精确 offline replay。

## 假设

如果 Stage250 historical 已保存或能唯一重建 prepare 的每 tick target、q/q̇ 和 contact/history，就可以只增加真实 prepare history，做一个 direct A/B；若任何关键输入缺失，则该 A/B 会同时猜测初态、命令和接触，违反单变量原则。

## 干预

无物理干预。只读检查：

- `stage250_video_straight.json` 的 10 行 prepare；
- `stage208_official_mujoco_adapter.py` 当前 prepare 与 publish 逻辑；
- official native event-v2 20 s；
- official reset-prefix event-v2；
- official activation-prefix event-v2。

## 对照

```text
可执行 prepare-history probe 所需：
start_q + 每tick target/Kp/Kd + 每tick actual q/dq
+ command应用physics-step + contact/solver history

现有 Stage250 prepare：
root pose/velocity + 50Hz wall dt + 聚合measurement skew/callback age
```

## 结果

### 1. Stage250 prepare 不能唯一重建

10 行 prepare 覆盖 `0.00–0.18 s`，但字段计数为：

| 字段 | 有效行 / 10 |
|---|---:|
| 31D joint q | 0 |
| 31D joint dq | 0 |
| command / physical target | 0 |
| ctrl / actuator force | 0 |
| contact / constraint | 0 |
| obs93 | 0 |
| action15 | 0 |

当前源码可见的公式是：

```text
alpha = smoothstep(elapsed / prepare_seconds)
target = prepare_start_q + alpha * (default - prepare_start_q)
```

但 historical 没有保存 `prepare_start_q`，也没有把当前源码 SHA 绑定到这次旧 capture。更关键的是，即使强行假设 start_q 等于 nominal，仍缺 actual q/dq、命令到达的 physics step、contact/constraint 与 warmstart；这不是一个可以通过插值公式补齐的数据缺口。

能确定的聚合时序只有：

- control dt p50 `20.037 ms`，p95 `20.184 ms`；
- measurement skew p50 `0.040 ms`，p95 `1.964 ms`；
- callback age p50 `1.638 ms`，p95 `1.998 ms`。

这些数值不能反推出四组 command 何时被 closed simulator 应用。

### 2. event-v2 能量化 recorder 侧时序

三份 event-v2 都保存了：

- 每条 command topic 在独立 recorder subscriber 的 monotonic receipt；
- 每个 50 Hz snapshot 所使用的 leg/waist/arm/head 最新 callback receipt；
- IMU 与 odom 最新 receipt；
- 完整 command event 数值和 group-local index。

较干净的 official-native 20 s 记录中：

- snapshot interval p50 `19.982 ms`，p95 `20.808 ms`；
- 四组 state receipt skew p50 `0.761 ms`，p95 `1.220 ms`；
- snapshot 时 leg/waist/arm/head state age p50 分别为 `1.118/0.860/0.597/0.367 ms`；
- IMU/odom age p50 为 `1.212/1.189 ms`；
- leg state receipt 距此前 recorder command receipt p50 `1.720 ms`，距下一条 p50 `0.130 ms`。

reset-prefix 和 activation-prefix 给出相近的常态范围，但偶有 `30–50 ms` host scheduling 尾部。三份数据的 command header populated fraction 都是 `0`，所以唯一可信的只是 recorder callback wall time。

### 3. event-v2 仍不能精确重放 Stage250

关键边界：

1. event-v2 的时间是**另一个 subscriber 收到消息**的时间，不是 closed simulator subscriber 消费命令、更不是 `mj_step` 应用 ctrl 的时间；
2. event-v2 使用 SDK 自带 native controller，command 约每组 500 Hz 重发；Stage250 adapter 是 50 Hz publisher，不能直接移植其 offset；
3. 50 Hz snapshot 只保存各 topic 的最新状态，不保存全部 1 kHz state callback 序列；
4. 没有 per-substep contact、constraint、warmstart 或 simulator-side applied-step index；
5. 同 group-index 的跨 topic receipt skew仅是 recorder 对齐近似，启动时可能存在 index offset，不能当控制周期真值。

因此可以按 recorder receipt 做“近似事件回放”，但不能宣称是 Stage250 closed physics 的 exact offline replay。

## 结论

`BLOCKED_BY_UNRECORDED_PREPARE_AND_APPLICATION_TIMING`

Phase30 已排除 MuJoCo 版本；Phase31 又确认，现有文件无法把剩余两个候选拆开：

- prepare 后隐藏的 contact/constraint/warmstart history；
- closed ROS command receipt/application 与 telemetry sampling 时序。

本阶段没有启动 physics probe，因为任何 direct prepare 都必须先猜 `prepare_start_q`、actual q/dq、应用时序或 contact state，最终结果无法证伪单一假设。

## 下一步

未来最小 capture 必须同时保存：

1. state-ready 时 full `mjData` integration snapshot，或至少 qpos/qvel/qacc_warmstart/efc state；
2. 每个 prepare tick 的 31D actual q/dq、最终 target/Kp/Kd；
3. simulator-side 四组 command receipt 与 applied physics-step index；
4. stand `t=0` 的 contact/constraint 或完整 integration snapshot；
5. adapter、scene、runtime config 的 capture-time SHA 绑定。

在此之前不解锁训练，不用 event-v2 的 recorder receipt 冒充 simulator application timing。

## 产物

- 机器可读结果：`reports/official_x2/phase31_prepare_and_ros_timing_audit.json`
- 审计工具：`tools/official_x2/audit_phase31_prepare_and_ros_timing.py`
- 纯测试：`tests/test_phase31_prepare_and_ros_timing.py`
- 测试结果：`4 passed`
