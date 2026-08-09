# X2 BASE Phase6：物理快照与 suffix replay 契约

日期：2026-08-09

训练更新：`0`

真机：未使用
结论状态：`DQ_RESTORED_PROJECTED_SNAPSHOT_EXACT_OLD_SUFFIX_UNAVAILABLE_TRAINING_LOCKED`

## 结论先行

Stage335 的 joint velocity 丢失已经定位并修复：不是 MuJoCo/Isaac 动力学差异，也不是 manager reset 或 `sim.forward()` 改写，而是 IsaacLab 首轮 reset 时 `soft_joint_vel_limits` 尚未初始化、全为零，原代码因此把所有 `dq` 静默 clamp 成零。修复后，16-env zero-update probe 的 requested/cache/PhysX/reset-return `dq` 对 source 投影值误差均为 `0.0 rad/s`。

但本阶段仍**没有**证明 deterministic suffix replay：90 条 source 中有 6 条 left-shoulder-roll 比官方下限最多越出 `0.000228 rad`，而旧 Stage326 的 5 份 trace 共 `3550/3550` 行都没有保存 controller state。当前只允许称为“投影后的物理/逻辑 snapshot 精确 + 新 controller serialization 入口就绪”，recovery PPO 与长训继续锁定。

## 假设

Stage335 reset-return 的 `dq` 差异来自 reset 生命周期中的一个明确写入/读回契约错误；若修复后物理 snapshot 成立，再建立 brake、latch、slew、heading 与 previous-action 的可序列化契约，作为未来 deterministic suffix replay 的必要条件。

## 干预

### 1. 单次 reset 边界测量

同一批 16 个状态在以下三个位置同时读取 Isaac cache 与 PhysX：

1. `write_joint_state_to_sim()` 刚返回；
2. manager reset 完成、logical finalizer 之前；
3. finalizer 完成、`sim.forward()` 之前。

没有训练、没有改变 PD、没有扫参数。

### 2. 首轮 velocity-limit 初始化修复

`ArticulationData.soft_joint_vel_limits` 初始化为零，只有第一次 `Articulation._apply_actuator_model()` 才写入 actuator limit，而它发生在首轮 reset event 之后。修复规则为：

- 正常 soft limit 已初始化时，继续使用 soft limit；
- 仅对 `<=0` 的未初始化 entry，回退到已经就绪的 URDF/PhysX hard velocity limit；
- hard limit 也非正时立即报错；
- 不修改 source 数据和 actuator 参数。

### 3. Controller snapshot 最小入口

新增 `aimdk_x2_stage208_controller_state_v1`，覆盖影响下一控制 tick 的状态：

- 三个 policy slot 的 previous/raw-issued action；
- step clock、move/stop 初始化状态；
- heading target/origin 与 heading hysteresis；
- lateral recovery state 与 2D slew bias；
- stop latch、emergency latch、hold targets；
- upper target/slew/fallback；
- state-prediction history；
- last move targets 与 prepare state。

Snapshot 强制绑定：

- 物理 state SHA-256；
- 完整 runtime args SHA-256；
- `clock_mode=step`。

物理 hash 或运行配置不一致时拒绝恢复。adapter 只新增纯 `export_controller_state()` / `restore_controller_state()`，未修改 action inference、symmetry 或 policy 输出。

## 对照

- 修复前：同一 16-env、seed 47、official-limit factor 1.0。
- 修复后：只改变未初始化 joint velocity limit 的 fallback。
- `recovery_fraction=0` 仍由单测证明严格 no-op。
- 旧 trace 审计只读，不把 summary/单帧 observation 推断成不存在的 controller state。

## 结果

### Reset 边界证据

修复前：

- source/expected `|dq|max = 1.6970097 rad/s`；
- write API 实际收到 `|dq|max = 0.0`；
- event-after-write 的 cache 与 PhysX 均为 `0.0`；
- manager/finalizer/forward 不是丢失位置。

修复后：

- write requested `|dq|max = 1.6970097 rad/s`；
- event-after-write cache/PhysX 对 expected 最大误差 `0.0`；
- manager reset 后最大误差 `0.0`；
- finalizer 后最大误差 `0.0`；
- reset-return joint velocity 最大误差 `0.0`；
- 16/16 状态执行一个 zero-action step 后存活；
- 93D observation、15D action、root pose/velocity、clock、command、previous/effective action 均保持原有精度。

证据：`/tmp/x2_recovery_phase6_joint_velocity_fix_v4.json`。

### 剩余物理差异

- runtime 对“合法投影后的 q”误差：`0.0 rad`；
- raw source 对 official hard limit：6/90 条 left-shoulder-roll 越限；
- 最大修正：`0.0002281107 rad`；
- 因此 probe 总裁决仍是 `passed=false`，没有放宽门禁伪造通过。

### Suffix 证据

- controller snapshot round-trip：通过；
- identical restore no-op：通过；
- physical hash mismatch：拒绝；
- runtime args mismatch：拒绝；
- wall-clock snapshot：拒绝；
- 旧 Stage326 trace：5 × 710 = `3550` 行；
- 有合法 controller state 的旧行：`0`；
- 缺失 controller state 的旧行：`3550`；
- deterministic suffix replay：**未执行、不可声称通过**。

### 测试

```text
19 passed in 1.07s
```

覆盖：controller snapshot、recovery reset、stop recovery state、official runtime import/entry contract。

## 结论

假设第一部分成立：`dq` 问题是一个确定的 IsaacLab 首轮初始化顺序 bug，已修复，且不是动力学不可迁移的证据。第二部分只完成了基础设施：新 trace 以后可以诚实保存/恢复 controller state，但旧 trace 没有这些信息，不能倒推出 exact suffix。

训练继续锁定，因为两个门尚未同时通过：

1. raw source physical snapshot 与 official X2 limit 的预注册等价门；
2. 同一 physical + controller snapshot 的 source suffix / restored suffix 确定性等价门。

## 下一步

1. 对 6 条 `0.000228 rad` 微越限明确二选一：重新从 official trace 导出合法 q，或预注册一个跨仿真 position tolerance；不得静默放宽。
2. 重新录一条短 matched-event official trace，每个候选 reset row 同时保存完整 physical snapshot、controller snapshot 及两者 hash。
3. 在同一 official runner 增加“从 snapshot 启动”的显式入口，比较未来 10–50 tick 的 q/dq/root/action/controller event；旧 trace 不参与该裁决。
4. 只有 physical gate 与 deterministic suffix gate 都通过，才允许将 `recovery_fraction>0` 接回独立 stand/recovery 训练。
