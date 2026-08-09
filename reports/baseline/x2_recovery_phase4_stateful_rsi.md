# X2 BASE Phase4：Stateful Recovery RSI 零更新契约

日期：2026-08-09
状态：**逻辑快照恢复通过；source→Isaac 物理等价失败；完整事件续接受阻；训练继续锁定**

> Phase6 更正（2026-08-09）：本报告中的 joint-velocity 阻塞已经定位为 IsaacLab 首轮 reset 时 `soft_joint_vel_limits` 尚为零而导致的静默清零，并已修复；修复后 reset-return `dq` 误差为 `0.0 rad/s`。当前阻塞收窄为 6/90 条 left-shoulder-roll 最大 `0.000228 rad` 的 raw-source 微越限，以及旧 trace 完全缺失 controller snapshot。以 [Phase6 报告](x2_recovery_phase6_snapshot_suffix.md) 为当前裁决，训练仍锁定。

## 一句话结论

Stage335 的 90 个状态可以精确恢复 episode clock、raw/effective action、gait phase、当前 command、root pose/velocity。Phase4 当时还观察到 reset-return joint velocity 不等价；Phase6 已证明它来自首轮 soft velocity limit 初始化顺序并修复。raw source 仍有 6 条极小 position 越限，且旧 trace 不具备 suffix replay 所需的 controller state，因此依旧不允许开始 PPO。

## 假设

只恢复 `q/dq/root` 不是合法的 recovery RSI。IsaacLab 的 reset event 会先运行，随后 action/command manager 和 episode clock 又被清零；如果不在 manager reset 之后恢复逻辑状态，策略看到的 93D observation、步态相位和实际执行目标会互相矛盾。

## 干预

- 新增 post-manager-reset 的 `StatefulRecoveryRLEnv` finalizer。
- 用 `source_index + trace_index` 将 Stage335 NPZ 严格回连到五条带 SHA-256 的官方 Stage326 trace。
- 恢复 episode clock、policy previous raw action、上一控制周期 issued/effective action、velocity command、standing/heading mask、command timer 和有界 force-moving latch。
- gait observation 和 gait-template action 共用同一个 moving latch；不修改旧 Gear-SONIC 文件。
- `recovery_fraction=0` 不产生 pending payload，finalizer 严格 no-op。

涉及文件：

- `tools/official_x2/recovery_reset_curriculum.py`
- `tools/official_x2/stateful_recovery_isaac.py`
- `scripts/probe_x2_stateful_recovery_reset.py`
- `tests/test_recovery_reset_curriculum.py`

本阶段没有 PPO、长训或 checkpoint 写入，也没有触碰 WBT、Stage222–225、Git 或百度网盘。

## 对照

`recovery_fraction=0`：继承原 reset 路径，不改变 episode clock、action、command，也不产生 stateful pending payload。纯单测确认该路径逐 buffer 不变。

## 数据契约审计

固定输入：

- Stage335 NPZ：90 状态，SHA-256 `4d8ce06b...5013`。
- 官方 source report：SHA-256 `b4755acd...f926`。
- 类别：eventual pass 63，eventual fail 27；五条 trace 分布为 21/8/21/21/19。
- 所有 90 行都通过 foreign-key、时间戳和 93D observation slice 的逐值核对，最大误差为 0。
- 35 行处于 moving gait phase，55 行处于 stationary phase。
- 12 行的 command 范数已经不大于 0.1，但 source gait phase 仍是 moving。若只用 command threshold 判相位，这 12 行会被静默改错，因此必须恢复 force-moving latch。
- episode clock 对 gait phase 的最大重建误差小于 `1e-5`；previous issued action 无缺失。

## 结果

### 纯测试

```text
9 passed in 1.03s
```

覆盖数据哈希、90-state contract、balanced sampling、物理 reset、fraction=0 no-op、source sidecar 无损 join、clock→phase 重建、逻辑 buffer finalizer 和无 pending no-op。

### 16-env IsaacLab zero-update probe（加强复核）

- observation：`16 × 93`；action：15D。
- episode clock、command、policy previous action、manager action、action-term raw/effective action：最大误差均为 0。
- gait phase 最大误差：`6.44e-7`。
- root position、root linear/angular velocity 误差为 0；root quaternion 的 `1-|dot|` 为 `1.79e-7`。
- 初次加强复核发现，通用 `soft_joint_pos_limit_factor=0.9` 会收缩 X2 的非对称 shoulder-roll 范围，使官方/default 的 `0 rad` 本身落在软限位外；当时 90/90 状态被投影，最大 `0.15293 rad`。
- 单变量改为官方 URDF/MJCF 硬限位语义（factor `1.0`）后，位置投影只剩 6/90 个 left-shoulder-roll 状态，最大 `0.000228 rad`，属于 source 略越过官方下限的微小残差。这个修正确实消除了主要位置契约错误，但没有解锁训练。
- source joint velocity 没有触发静态 soft-limit clamp，但 reset 返回时 31 个关节速度均出现差异，最大为 right-knee `1.69701 rad/s`。因此先前只检查 q/root-z 的 probe 高估了物理等价性。
- observation 全部 finite。
- 随机样本中包含 3 个“低 command 但强制 moving”的困难状态。
- 执行一次 zero action 后 16/16 环境仍存活。
- PPO update：0。

因此，**逻辑 buffer 与 root 状态门通过，但完整 source snapshot 等价门失败**。16/16 一步存活只证明投影后的状态不会立即终止，不能替代等价性门。

### 官方 matched-event 不训练门禁

五条 immutable Stage326 source trace 全部来自 `aimdk_x2_v1_official_mujoco`，共享：

- 50 Hz 控制；
- `official_kp_ankle × 1.2`；
- Stage208 deterministic 93D observation；
- 15D lower/waist residual + gait template action contract。

其中 full gate 为 3/5，Stage335 的 63/27 eventual label 与这些官方 continuation 完整对应。这证明数据具有官方 matched-event provenance。

但这只是 observational gate。它**没有**证明新官方进程从某一行 mid-event reset 后能重现原 suffix。普通冻结 checkpoint 从 episode 开头再跑一遍，也不能回答这个问题，所以本阶段没有拿无信息增益的 full-episode rerun 冒充 RSI 验证。

## 阻断原因

现有 NPZ/trace 已保存：物理快照、当前 command、gait phase、相邻 issued action。尚未保存足以确定未来闭环的完整状态：

- `_brake_command()` 后续反馈调度状态；
- lateral supervisor 的 latch/slew/internal state；
- official runner 可恢复的完整 controller event state。

因此只能精确恢复 snapshot 和短时 moving latch，不能保证完整 stop-event continuation 的因果等价性。

## 结论

**假设只部分成立。** 先前 reset 确实缺少关键逻辑状态，Phase4 已将这些 buffer 精确恢复且 fraction=0 无侵入；复核同时定位并基本消除了 0.9 软限位造成的 shoulder-roll 大投影。但 6 个微小越限状态、reset 后 joint velocity 差异以及 official suffix 内部状态仍未解决，所以 recovery training 必须保持锁定。

## 下一步

1. 保留官方硬限位语义，处理剩余 6 个微小 shoulder-roll 越限状态，并定位 reset 后 joint velocity 丢失发生在哪一层。
2. 在官方 matched-event 采集器中同步序列化 brake、latch、slew 和 supervisor/controller state。
3. 新增隔离的 official mid-event restore runner，恢复完整 qpos/qvel 与控制器状态。
4. 对同一 state 比较 source suffix 与 restored suffix；只有通过确定性等价门，才把 `recovery_fraction>0` 接入训练入口。
