# BASE Phase19：Recorder v2 与三角色采集门

> 日期：2026-08-09
> 裁决：**v2 数据真实性门通过，但三角色库存门未通过；16-env 注入和训练继续锁定。**
> 机器可读结果：[x2_recovery_phase19_recorder_v2_capture.json](x2_recovery_phase19_recorder_v2_capture.json)

## 假设

Phase18 的物理状态注入是精确的，但 93D 在 success/height-fail 上不等价。代码级原因已经定位为：

- 接触位使用精确浮点 gait clock，而旧 sidecar 只保留可被取整/舍入的外部时间；
- actor 的角速度和 gravity 来自 torso IMU topic，旧 sidecar 却只有 root/odom 物理状态；
- previous/issued action 必须按 actor 推理前与物理执行后的边界分别保存。

本阶段假设：若 recorder 原样保存 actor 真正消费的 93D 及其独立来源，就能消除这类“从 root 反推 IMU、从整数时钟反推 contact”的伪等价；随后只有 success、critical、height-collapse 三类都齐全，才允许做 zero-update injection。

## 干预

新增默认关闭的 recorder schema v2，逐行保存：

- torso IMU 原始 frame、姿态、角速度、加速度和 covariance；
- odom quaternion 与世界线速度源；
- 31DOF q/dq/default joint order；
- predictor 调用前后状态与 cache step；
- 精确浮点 phase input、offset、period、double-support fraction、contact 输出和 latch；
- pre-inference previous action、实际 93D、完整 93/121/123D model input；
- post-inference controller/physical/action-history 与逐层 hash。

每行都从这些独立来源重建 93D，要求绝对最大误差 `<=1e-6`。actual action 仅是 history/state，不是 expert imitation label。

随后冻结 Phase17 的 Stage326 mixed-outcome 控制合同，最多运行 5 个 official AimDK MuJoCo episode；模型、PD、stop controller、preview、时钟与 upper-body 固定均不改变。0 PPO、0 optimizer、0 真机。

## 对照

- moving：Stage306，SHA `da95011f…bc4c`；
- stationary：source i150，SHA `edb73c7c…7565`；
- adapter：`6b7c6c35…f9d9`；
- v2 contract：`eae96b1e…dbdb`；
- `brake_blend_to_policy`、official PD stiff1.2、step clock、fixed upper；
- 目标库存：至少 `1 full-success + 1 critical + 1 height-collapse`，达到即停；任一行自一致失败也立即停。

旧 Phase17 文件没有重采、覆盖或改写。Phase11 的历史采集 SHA 仍保留，只把报告中的 current-source provenance 更新为 Phase19 live adapter，并明确 source drift。

## 结果

### 1. Schema v2 本身通过

- focused tests：`13 passed`；
- 相关回归：`96 passed`；
- 有效 sidecar：4 条 × 201 行 = `804` 行；
- 最坏 93D 重建误差：`3.5763e-7 < 1e-6`；
- 804 个 snapshot hash 全部唯一；physical-state hash 为 `802/804` 唯一；
- torso IMU 角速度与 root angular velocity 在有效行中确实可以不同，且 actor 93D 始终由保存的 IMU 源逐项重建，证明没有再用 root 值覆盖 IMU。

### 2. 五次物理预算没有凑齐三类

| Episode | 裁决 | v2 rows | Max 93D error | Stop z min | Stop tilt max | Stop drift |
|---|---|---:|---:|---:|---:|---:|
| r1 | invalid artifact | 0 个可用 | 不分类 | 不使用 | 不使用 | 不使用 |
| r2 | height-failure | 201 | `2.38e-7` | 0.139 m | 1.545 rad | 0.573 m |
| r3 | height-failure | 201 | `3.58e-7` | 0.139 m | 1.529 rad | 0.572 m |
| r4 | full-success | 201 | `1.19e-7` | 0.637 m | 0.230 rad | 0.147 m |
| r5 | full-success | 201 | `1.19e-7` | 0.622 m | 0.213 rad | 0.111 m |

最终角色计数：

```text
success        2
critical       0
height-failure 2
invalid        1
```

r1 的物理 episode 和 201 次行内检查已经运行完，但最终 sidecar 封装把完整 deploy summary 送入 `allow_nan=False` hash；其中禁用的 upper threshold 是 `Infinity`，因此 sidecar 未落盘。修复只把 hash-bound outcome 投影为与 v1 一致的有限布尔 gate。r1 trace SHA 为 `5642a508…da63`，已保留且没有重跑/覆盖；由于没有完整 sidecar，它不参与类别或库存。

### 3. 预注册停止条件生效

critical v2 episode 缺失，所以：

- 没有构建伪三角色 manifest；
- 没有启动 16-env zero-update injection；
- 没有给出 5-update 解锁结论；
- 没有训练，也没有用 success/failure action 当 expert label。

## 结论

Phase19 解决了 Phase18 最关键的**观测来源真实性**问题：新的 sidecar 能从真实保存的 torso IMU、精确 phase/contact generator、q/dq、predictor 和 previous action 独立重建 actor 93D，而不是事后拼一个“看起来像”的向量。

但它没有完成三角色 reset curriculum。五次物理预算里出现了两个稳定成功和两个明确坍塌，却没有临界结果；缺一类就不能测试 role-aware 16-env 等价，更不能据此开 5-update。

因此裁决是：**schema 门通过，inventory 门失败，训练继续锁定。**

## 下一步

唯一合理的后续是另行预注册一次“只补 critical v2 库存”的采集任务，仍使用相同 hash、自一致阈值和无训练边界。拿到至少一条完整 critical v2 sidecar 后，才可重新做 success/critical/height-fail 的 16-env zero-update 注入；三类全过之前不得启动 5-update。

## 关键文件

- v2 contract：`tools/official_x2/stop_event_suffix_contract_v2.py`
- adapter：`tools/official_x2/stage208_official_mujoco_adapter.py`
- runner：`tools/official_x2/run_phase19_stage326_v2_capture.sh`
- validator：`tools/official_x2/validate_phase19_stop_event_episode_v2.py`
- tests：`tests/test_phase19_stop_event_suffix_contract_v2.py`

## 边界

- official 指 AimDK v1 MuJoCo/ROS，不是真机；
- 没有真实足底六维力、COP 或 GRF；
- sidecar 是 stateful evidence，不是 vendor ROS 无损 mid-event restore 证明；
- 没有 WBT、Git、百度网盘或真机操作；
- 训练保持锁定。
