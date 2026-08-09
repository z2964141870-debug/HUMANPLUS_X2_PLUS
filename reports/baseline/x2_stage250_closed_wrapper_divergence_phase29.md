# BASE Phase29：Stage250 closed AimDK 与 Phase28 direct MuJoCo 最早分叉审计

日期：2026-08-09
状态：`BLOCKED_BY_CLOSED_WRAPPER`
训练：锁定；本阶段没有训练、没有新回放、没有控制器修改、没有真机操作。

## 游戏任务卡

- [x] 冻结 Stage250 historical full-gate trace 与 Phase28 唯一 direct failure cache。
- [x] 对齐 stand `t=0` 后的 `0–0.2 s`，比较 qpos/qvel/root/action 与可得的加速度证据。
- [x] 审计官方 launcher、二进制、MuJoCo module、runtime config、reset/default 顺序和编译来源。
- [x] 明确 contact、ctrl、actuator force、warmstart、solver state 哪些可比、哪些缺失。
- [x] 只在能唯一锁定单变量时才运行 matched probe。
- [ ] 唯一根因：未解锁；现有证据同时存在至少三个独立域差异。

## 假设

Stage250 在 closed AimDK ROS 中 full-gate，而 Phase28 在 direct official-MJCF 中从 stand 即坍塌，可能不是 actor/reference 本身退化，而是两条执行路径在第一个物理步就不等价。候选包括：

1. MuJoCo runtime/compiler 版本；
2. prepare 结束时未保存的 contact/solver warmstart/history；
3. closed ROS 的异步 sensor/command scheduling 与 direct synchronous stepping。

## 干预

无物理干预。只读使用：

- Stage250 straight historical trace：prepare 10 行 + 700 行完整 93D/action trace；
- Phase28 14,000 行、1 kHz substep cache；
- Phase28 700 行、50 Hz control cache；
- 官方 `scene.xml/x2.xml/default.yaml`、launcher、resolved runtime config；
- stripped `aima-sim-app`、`libaima-sim-module-mujoco.so` 的 ELF/Dynamic symbol/只读反汇编证据。

对齐规则固定为：historical trace index 10 是 stand `t=0`；Phase28 从该行重建同一可见 qpos/qvel，`mj_forward` 后开始 stepping。其后 direct 第 `20*k` 个 1 ms step 与 historical 第 `k` 个 20 ms state 对齐。

## 对照

```text
Stage250 historical
closed aima-sim-app + ROS callbacks + vendor MuJoCo 3.3.7
reset/default → 0.2 s prepare → stand

Phase28 direct
Python MuJoCo 3.4.0
重建 stand t=0 可见 qpos/qvel → mj_forward → 20×1 ms synchronous step
```

两支在 stand `t=0` 的 93D observation 与 action contract 已匹配：初始 action 最大误差仅 `7.45e-8`。这排除了“第一个 action 映射不同”作为分叉起点，但不等价于完整物理状态相同。

## 结果

### 1. 分叉发生在第一个 20 ms 内

在 `t=0.02 s`：

- root position L2 error：`0.00252 m`；
- root quaternion geodesic error：`0.02758 rad`；
- root angular velocity max error：`2.518 rad/s`；
- joint q max error：`0.04166 rad`，首要关节为 left hip pitch；
- joint dq max error：`4.438 rad/s`，首要关节仍为 left hip pitch；
- 下一次 actor action max error：`0.4184`。

固定诊断阈值（不是 gate）的首次越界：

| 指标 | 首次越界 |
|---|---:|
| joint dq max > 0.5 rad/s | 0.02 s |
| root orientation > 0.05 rad | 0.04 s |
| joint q max > 0.05 rad | 0.04 s |
| root position L2 > 0.01 m | 0.06 s |

因此事件顺序是：**物理速度先分叉 → 下一次 observation/action 随之分叉 → 姿态和位置进一步失真**。不能把 20 ms 后的 actor action 差异当成最初原因。

首个 1 ms direct acceleration 与 historical 首 20 ms 平均 acceleration 的比较显示，最大差异集中在 left/right hip pitch、waist pitch、left knee。由于采样带宽不同，这只定位最早受影响的自由度，不是等带宽因果证据。

### 2. runtime 版本并不相同

- closed module 动态链接：`libmujoco.so.3.3.7`；
- Phase28 Python direct 实际 runtime：`MuJoCo 3.4.0`；
- official XML timestep：`0.001 s`；direct compiled model 为 `nq=38, nv=37, na=0, nu=31`。

`na=0` 说明该 official motor model 没有 MuJoCo actuator activation state；“隐藏 activation”不是候选。但 3.3.7/3.4.0 的 contact/solver 数值差异仍未被控制。

### 3. direct 没有恢复 prepare 的完整积分状态

historical prepare 从约 `root z=0.6796 m` 运行到 stand 初始 `0.6523 m`，保留了真实的双脚接触历史。Phase28 只恢复最终可见 qpos/qvel，然后 `mj_forward`：

- 没有 closed wrapper 的 `qacc_warmstart`；
- 没有 constraint/efc solver state；
- 没有此前 contact impulse/history；
- 没有此前 ctrl/command-arrival state。

Phase28 首个 substep 已检测到左右脚、20 个 contact，但 historical trace 没记录 contact 或 force，无法证明两支的接触约束解相同。

### 4. ROS scheduling 也不是 direct stepping

官方 resolved config 使用：

- joint state 1 kHz；IMU/odom 500 Hz；
- joint command subscriber executor 5 threads；
- ROS best-effort channel；
- simulator `mj_step` 与 publisher/subscriber 异步运行。

historical 前 11 个有效控制状态中：

- control wall dt mean `0.0200007 s`；
- measurement skew p95 `0.001968 s`；
- callback age max p95 `0.001975 s`。

Phase28 则严格同步：20 次 `mj_step` 后才做下一次 inference。closed wrapper 内 command 实际落在哪个 1 ms substep没有记录，因此这也是独立未控变量。

### 5. 可观测性边界

| 字段 | historical closed | Phase28 direct | 可直接归因？ |
|---|---|---|---|
| 50 Hz qpos/qvel/root/obs/action | 有 | 有 | 是 |
| qacc | 仅 50 Hz finite diff | 仅 1 kHz finite diff | 只作诊断 |
| contact / `mj_contactForce` | 无 | 有 | 否 |
| ctrl / actuator force / qfrc_constraint | 无 | 有 | 否 |
| actuator activation | 无 | `na=0`，结构上不存在 | 排除该候选 |
| qacc_warmstart / efc solver state | 无 | Phase28 未记录 | 否 |
| compiled `mjModel` dump | 无 | 可重编译 | 否 |
| command 到达的 physics substep | 无 | synchronous | 否 |

historical 为录制画面挂载了 `scene_report.xml`，与 vendor scene 的语义差异只有 camera `<statistic>`，physics include、floor、`x2.xml` 未变；但 closed compiled `mjModel` 本身没有落盘，所以仍不能做 byte-level compiled-model 等价声明。

## 结论

Phase28 的失败不能推翻 Stage250 能力，也不能证明 official MJCF 本身不稳定。它证明的是：

> 从同一个可见 q/q̇/93D/action 开始，并不足以复现 closed AimDK 的后续物理；分叉在第一个 20 ms 的 q̇/角速度已经发生。

目前不能诚实锁定一个根因，因为至少三项同时不同：MuJoCo 版本、prepare 后隐藏 solver/contact history、ROS command/sensor scheduling。故裁决为：

`BLOCKED_BY_CLOSED_WRAPPER`

没有启动新 probe，也没有解锁训练或 WBT native seed。

## 下一步（仅预注册，未授权执行）

最小单变量 probe：

```text
control:   Phase28 direct prefix, Python MuJoCo 3.4.0
candidate: 完全相同 prefix, Python MuJoCo 3.3.7
冻结: XML/state/action/PD/20×1ms stepping/ONNX
horizon: 0.2 s
比较: 每1ms与20ms的qpos/qvel/action/contact
```

若 3.3.7 仍与 historical 明显不同，只能排除“版本差异足以解释失败”；下一份真正有信息增益的证据必须来自 closed wrapper 在 stand `t=0` 的 full integration-state snapshot（含 warmstart/constraint/control timing），而不是继续调策略或 reference。

## 产物

- 机器可读结果：`reports/official_x2/phase29_closed_wrapper_divergence.json`
- 可复现工具：`tools/official_x2/audit_phase29_closed_wrapper_divergence.py`
- 纯测试：`tests/test_phase29_closed_wrapper_divergence.py`
