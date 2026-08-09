# BASE Phase28：Stage250 straight extended direct-official 资格重采

## 一句话裁决

历史 Stage250 的 observation/action 合同可以精确复算，重建状态也能做确定性 direct-MJCF 分叉；但唯一一条 extended straight 在 0.06 s 就与官方历史物理轨迹明显分叉，并在 stand 阶段倒地，所以它是明确的 **direct-vs-AimDK domain mismatch**，不能晋级为 native dynamic seed，也不能解锁后续 WBT runner。

## 游戏任务

- [x] 冻结未修改的官方 `scene.xml`、Stage219 ONNX、stand ONNX、template、default、PD、50 Hz 合同。
- [x] 用历史 700 帧 obs 顺序复算 ONNX raw、第一次 clip、template、最终 clip/action。
- [x] 从第一条完整 Stage250 stand 状态重建 qpos/qvel/root，并做双 fork 未来 10 tick exact。
- [x] 只运行一次 straight；没有调参、重试或第二条 rollout。
- [x] 记录 700 个 control tick 和 14,000 个 1 kHz physics substep。
- [x] 每个 substep 保存 qpos/qvel/root、contact geom/position/frame/wrench、ctrl/actuator force/qfrc、actor/action/target 全链路。
- [x] 用 `mj_contactForce` 审计 realized contact/slip/clearance/GRF provenance。
- [ ] 复现 Stage250 nominal stand/start/move/stop/full gate。
- [ ] native dynamic seed 资格。

## 假设

如果 Stage250 trace 中的 93D observation、15D action、root 字段足以恢复官方物理状态，那么相同的官方 scene、actor、template、PD/default 与 50 Hz action contract 应在 direct MuJoCo 中复现 nominal full gate；同时每物理子步 contact force 可为原生动态 seed 提供比 Phase27 离线逆动力学更直接的模型证据。

## 干预

没有策略或物理参数干预。唯一变化是执行位置：

```text
历史：AimDK ROS wrapper → 官方 MuJoCo
Phase28：同一官方 scene.xml → project-side direct MuJoCo
```

初态取历史第一条完整 stand row，由 93D q/dq、root xyz/yaw 与 projected gravity 重建。episode 固定为：

```text
stand 100 ticks → move 200 ticks → stop 400 ticks
50 Hz controller / 1 kHz physics / vx=0.30 m/s
```

## 对照与 preflight

### 1. 历史 action contract

对历史 700 帧顺序维护 main/stationary previous-action、move-start heading rebase、stop handoff、lateral supervisor：

| 项目 | 结果 |
|---|---:|
| decoded rows | 700 |
| historical previous-action 最大误差 | `2.98e-7` |
| historical final action 最大误差 | `2.98e-7` |
| ONNX raw actor 最大绝对值 | `4.400` |
| first clip 最大绝对值 | `1.000` |

这证明 Stage250 的修复合同被正确复现：raw 可超过 1，但历史 last-action 与执行前 residual 都遵循第一次 clip；随后 template/supervisor 组合再做最终 clip。

### 2. 同状态 deterministic fork

从同一重建初态分叉两个 direct runner，未来 10 tick：

| 字段 | 最大误差 |
|---|---:|
| qpos | 0 |
| qvel | 0 |
| obs93 | 0 |
| final action | 0 |
| stage event | 完全相同 |

因此结果不是 direct runner 自身随机或 snapshot 不确定性。

## 唯一 rollout 结果

### Nominal gate

| gate | 结果 | 主要连续指标 |
|---|---:|---|
| stand | **FAIL** | root z min 0.077 m；tilt max 1.710 rad；drift 1.277 m |
| startup | **FAIL** | root z min 0.072 m；tilt max 1.672 rad；forward -0.145 m |
| move | **FAIL** | root z min 0.072 m；forward -0.166 m；heading error 3.137 rad |
| stop | **FAIL** | root z min 0.069 m；tilt max 1.884 rad；tail speed 0.846 m/s |
| full | **FAIL** | 四个子门均失败 |

机器人不是在正常走路后偶发失败，而是在 stand 阶段已经失稳：tick 0 的 direct obs93 与历史首帧逐项完全相等，首 action 误差仅 `7.45e-8`；但到 0.06 s，root tilt 与历史下一状态已相差 `0.0531 rad`。约 0.4 s 后 tilt/height 明显坍塌，后续 move/stop 已发生在倒地状态。

### Realized contact（失败域，仅作 provenance）

| 指标 | left | right |
|---|---:|---:|
| move contact tick fraction | 0.275 | 0.200 |
| contact-point slip p50 | 0.693 m/s | 0.594 m/s |
| contact-point slip p95 | 2.423 m/s | 2.601 m/s |
| normal force p50 | 99.4 N | 73.8 N |
| normal force p95 | 638.2 N | 565.9 N |
| DS→SS→DS cycles | 0 | 0 |

无接触时 sole clearance p95 为 0.138/0.255 m，但这来自已经倒地后腿部离地，**绝不能**当作有效 swing clearance。倒地后 drift 小、脚离地高或某些 contact force 存在，都不能算步态成功。

GRF provenance 比 Phase27 更直接：每个 1 ms substep 使用 `mujoco.mj_contactForce`，并保存 geom IDs、world position、contact frame 和 6D contact-frame wrench；`force × 0.001 s` 仅标作近似 impulse。它仍是官方 MJCF 的模型接触力，不是 X2 足底传感器或 closed ROS 独立真值。

## 记录完整性

外部 cache 已冻结：

- 14,000 行 substep gzip JSONL，约 16.2 MB；
- 700 行 control-tick JSON，约 3.43 MB；
- substep 字段含 full qpos/qvel、root pose/velocity、contacts、ctrl、actuator force、qfrc actuator/constraint、raw/clip/template/final action、PD target；
- repo manifest 保存绝对路径、行数、字节数和 SHA256，不复制大 trace。

## 结论

### 已证实

1. Stage250 的 ONNX/action/last-action 合同没有再错；
2. 重建 direct runner 是确定性的；
3. “同 obs + 同 action + 同 scene/PD”仍不足以复现 AimDK 历史物理闭环；
4. 因此 Phase27 的 50 Hz trace 只适合作 kinematic/control warm-start，不能直接提升为 dynamic seed。

### 尚未证实

不能仅凭这一条把差异归因于某一个参数。最小缺口是历史 trace 没有 AimDK 仿真器的完整 integration state，包括 contact solver warm-start、qacc/constraint state、可能的 actuator/controller internal state 与精确多 topic 同步边界。当前证据只证明这些未记录状态或 wrapper 语义中至少有一项是必要的。

### 资格裁决

```text
qualified_native_dynamic_seed = false
WBT dynamic-seed runner unlock = false
training unlock = false
```

## 下一步

按任务卡在失败处停止，不调合同、不重跑。若将来要继续，必须先获得/记录 AimDK 初态的 lossless `mjSTATE_INTEGRATION` 等价物，或在官方 wrapper 内直接增加同样的 substep recorder；在没有这一物理状态闭合前，继续调 direct runner 会把 domain mismatch 误当策略问题。

## 证据与边界

- manifest：`reports/official_x2/phase28_stage250_extended_direct_manifest.json`
- runner：`tools/official_x2/run_phase28_stage250_extended_direct.py`
- cache audit：`tools/official_x2/audit_phase28_extended_cache.py`
- tests：`tests/test_phase28_extended_direct.py`（2 passed）
- substep cache SHA256：`2581e8228425e39bbd4f228be32566c6488b9f28916ff8827520b9ae324ecaab`
- control cache SHA256：`9abc1574535c1f36ae36a094c0be77019ce6a5bdd5aaefc5aa0bc19af3b0f40a`
- 无训练、无第二次 rollout、无 adapter/WBT/Git/百度/真机修改。
