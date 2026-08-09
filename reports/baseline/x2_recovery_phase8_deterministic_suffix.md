# BASE Phase8：official-MJCF deterministic suffix 合同

## 结论先行

Phase6 的 controller snapshot 已从“字段看起来齐全”推进到一次可证伪的动态后缀验证：在**未修改的 X2 官方 `scene.xml`** 中，把 `mjSTATE_INTEGRATION` 物理态与 controller state 一起恢复到新的 runner 后，未来 **10 / 25 / 50 个 50 Hz tick** 的 `q / dq / root / action / controller event / signed pelvis pitch` 全部与不中断的 source 分支一致，最大误差均为 **0**。25 和 50 tick 已跨过 `stop_curriculum → stop_stationary` 的 latch/handoff，因此不是只验证静态站立。

这项通过的准确边界是：

> unchanged official MJCF + same-process test-only closed loop 的 deterministic suffix 通过。

它**不是**官方 AimDK ROS 二进制闭环的 mid-event restore 通过。官方公开接口没有完整状态注入能力，不能把本结果扩大表述为 ROS suffix closure，也不能据此解锁训练。

## 1. 官方接口审计

### 已检查

- 官方 simulator 配置：`x2_rl_deploy_mujoco/.../simulator/default.yaml`；
- 官方 README 的 joint / IMU / odom 接口表；
- closed subscriber library：`libaima-sim-module-subscriber.so.0.0.0`；
- official simulator executable / shared-library symbol 与字符串；
- Phase6 controller snapshot contract。

### 结果

- joint command、joint state、IMU、odom 都是普通运行时 topic；
- `/aima/sim/option/command` 的公开 `OptionCommandSubscriber::Handle` 只发现 `enable_imu` / `disable_imu`；
- GUI/README 有整体 reset，但没有公开 root pose、`qpos/qvel`、`ctrl`、warm-start、contact/plugin state 的注入 RPC/topic；
- closed binary 内部使用 MuJoCo state/history API 不等于它对项目开放了无损注入接口。

因此，继续搜索 ROS reset 参数没有信息增益。Phase8 改用隔离的 direct-MJCF test hook，并在证据里显式记录 `official_aimdk_ros_closed_loop=false`。

## 2. 假设

完整的 `mjSTATE_INTEGRATION` 与 Phase6 controller snapshot 足以唯一确定 matched-event controller 的未来后缀；如果仍发生分叉，第一次失败 tick 和字段将指出缺失的物理态或控制器态。

## 3. 干预

在 Stage351 的事件顺序下运行：

```text
prepare 0.2 s
→ stand 2.0 s
→ matched start/move 5.2 s
→ matched deceleration
→ stationary-policy handoff
```

快照边界预先固定为：

- `sequence_step = 460`；
- 下一 tick 为 stop elapsed `1.80 s`；
- 距 `stop_transition_seconds = 2.0 s` 的 policy latch 约 10 tick；
- source 分支不中断继续 50 tick；
- restore 分支新建 `MjData` 和 controller，再注入相同 physical/controller snapshot，继续 50 tick。

绑定内容：

- 官方 scene、main actor、stationary actor、gait template 的内容 hash；
- `mjSTATE_INTEGRATION` 的 379 个 float64；
- Phase6 controller schema 的 previous/issued action、sequence/control step、heading、stop latch、handoff、upper/预测历史等状态；
- step clock 和 runtime args hash。

关键 hash：

- asset manifest：`a27926e91ccacec0d9b39c27311681b91afde35825dee61bbedbf8d2812d729a`
- physical snapshot：`937387769f3490ec5ce5c34fa27d1d3c43ff1668ea41457fc79da8da2fc5aa6d`
- runtime args：`b9b4b205f04450ee14d59371197bd7c998c8dec8748c603eeaea34c52d73d62b`

## 4. 对照与预注册门

对照是同一 runner 在同一 snapshot boundary 不中断继续运行。门在运行前写死：

| 字段 | 门 |
|---|---:|
| snapshot physical integration state | bitwise exact |
| controller snapshot round-trip | exact JSON contract |
| future q / dq / root | abs max `≤ 1e-12` |
| future action | bitwise exact |
| controller event/latch | exact |
| signed pelvis/root pitch | abs error `≤ 1e-12 deg` |

比较 tick 固定为 10 / 25 / 50；任何未来 tick 首次失败即停止把结果解释成 suffix pass，并报告首先分叉字段。没有在看过结果后调 tolerance。

## 5. 结果

| Horizon | q max | dq max | root max | action max | event | pitch error | 结果 |
|---:|---:|---:|---:|---:|---|---:|---|
| 10 tick | 0 | 0 | 0 | 0 | exact | 0 deg | 通过 |
| 25 tick | 0 | 0 | 0 | 0 | exact | 0 deg | 通过 |
| 50 tick | 0 | 0 | 0 | 0 | exact | 0 deg | 通过 |

补充：

- physical snapshot restore：bitwise exact；
- controller export/restore/export：exact；
- tick 10 时 latch 仍为空；
- tick 25、50 时 `stop_hold_latch_s = 2.0`，说明比较确实跨过事件；
- `first_failure = null`。

## 6. signed sagittal posture（只报告，不进 full gate）

Phase7 约定负值为后仰。本 probe 每行保存 `root_pitch_deg`，并按阶段统计：

| 阶段 | mean | p05 | p95 |
|---|---:|---:|---:|
| stand | -6.81° | -7.54° | -3.83° |
| start（move 前 1 s） | -7.57° | -10.34° | -3.99° |
| move | -10.68° | -12.48° | -8.08° |
| stop | -7.97° | -11.91° | -2.58° |

这些值复现了 Phase7 的持续后仰表型，但本阶段没有官方 locomotion 自然姿态阈值，所以只作为相对 metric；它不改变 deterministic suffix 的 pass，也不加入 full gate。

## 7. 结论

1. Phase6 controller snapshot 的字段在这个隔离 runner 内没有发现动态缺口；
2. source-joint velocity reset-return 不等价不能再归咎于“controller history 一定没存完”：本次完整 physical integration state + controller state 可精确复现；
3. 旧 trace 仍然不能 suffix replay，因为它们没有绑定并保存这两类 snapshot；
4. 官方 closed ROS 的缺口仍真实存在：没有 mid-event full-state injection，就不能做同等级 ROS 分叉；
5. 本结果是测试基础设施甜点位，不是性能改进，也不是训练解锁门。

## 8. 下一步

- BASE 性能主线可用该 runner 做**单变量、同快照**的 paired suffix attribution，减少 reset 初态方差；
- 只有新 trace 显式写入 asset-bound physical/controller snapshot，才允许宣称可恢复；
- 如未来取得官方 simulator state injection hook，应原样复用 10/25/50 tick 契约在 ROS binary 上复验；
- 当前先停止扩实验；持续后仰应回到 Phase7 已定位的 sagittal target/controller contract，而非拿 suffix pass 代替姿态修复。

## 9. 产物与验证

- Runner：`tools/official_x2/run_testonly_official_mjcf_suffix_probe.py`
- 纯测试：`tests/test_official_mjcf_suffix_probe.py`
- 完整结果：`reports/baseline/x2_recovery_phase8_deterministic_suffix.json`
- 可恢复快照：`reports/baseline/x2_recovery_phase8_snapshot.json`
- 回归：Phase6 + Phase7 + Phase8 共 `13 passed`。

训练、真机、WBT、Git、百度网盘均未触发；vendor 文件未修改。
