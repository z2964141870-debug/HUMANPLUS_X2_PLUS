# BASE Phase22：Phase21 共同失稳事件顺序归因

> 日期：2026-08-09
> 性质：严格只读；15 条既有 official raw trace，没有训练、仿真、阈值调整或 adapter 修改。
> 裁决：**三组共享同一失败链，f005 没有改变顺序且把 tilt/height collapse 提前；倒地后的 drift 变小不是恢复。**
> 机器可读结果：[x2_recovery_phase22_common_instability.json](x2_recovery_phase22_common_instability.json)

## 假设

Phase21 的 f005 虽然 stop drift/settle mean 较小，却仍是 stop/full `0/5`。本阶段检验：

1. source、f000、f005 是否共享同一最早异常和失稳顺序；
2. f005 是否真正把异常推迟，还是只让倒地后的漂移变小；
3. 失稳首先来自交权 target jump、policy/history OOD、PD 跟踪、支撑时序、IMU/root，还是早在 recovery 接管前已存在的 heading/lateral 问题。

## 干预与对照

没有物理干预。对 Phase21 三组各 5 条 trace 按 stop 第 2.0 s recovery handoff 对齐。

沿用既有、未修改的判据：

- root z `0.45 m`、tilt/heading/lateral `0.30` official gate；
- Stage335 与 Phase14 相同的 robust median/IQR、group RMS、reference LOO nearest-neighbor p95；
- recorder-v2/Phase20 的 432 个 eligible exact actor 93D row，使用同一 LOO-p95 与 p01–p99 描述门；
- action/target/PD step 只问是否超过同 episode handoff 前 1 s 的最大值，不另设倍数。

`gait_phase[2:4]` 只能表示 actor 使用的 contact generator 状态，不能写成真实脚底接触、GRF、COP 或力传感器结果。

## 结果

### 1. heading/lateral 在 recovery handoff 前已经失败

| 首次越过 official 0.30 门，相对 handoff 中位数 | source | f000 | f005 |
|---|---:|---:|---:|
| heading | -4.56 s | -4.70 s | -4.48 s |
| lateral | -3.20 s | -3.66 s | -3.28 s |

每组 5/5 都在 recovery 接管前越界。因此：

- recovery actor 无法对 Phase21 的 move gate 负责；
- f005 训练只改 recovery slot，不可能修复已经发生的直行偏航/横漂；
- 这解释了三组 move 都为 `0/5`，但不能解释后续为何倒地。

### 2. Phase13 blend 已消除“handoff 首 tick 大跳变”作为共同根因

0.5 s blend 下，action/physical target 的一步变化超过 handoff 前 1 s max 只出现在：

- source 2/5，约 `+2.19 s`；
- f000 3/5，约 `+2.26 s`；
- f005 1/5，约 `+2.10 s`。

这些事件晚于 tilt，并接近或晚于 height collapse；不是 15 条共同的首发事件。PD target-to-q error 超过 pre-handoff max 同样只在 source 2/5、f000 3/5、f005 1/5 出现。现有 trace 不支持把“首 tick target jump”或“统一 PD 跟不上”继续当 Phase21 根因。

post-handoff 最大 target step 的 joint-group L2 中位数约为：

| 组 | source | f000 | f005 |
|---|---:|---:|---:|
| 左腿 | 0.223 rad | 0.279 rad | 0.220 rad |
| 右腿 | 0.298 rad | 0.277 rad | 0.310 rad |
| 腰 | 0.065 rad | 0.076 rad | 0.079 rad |

f005 的右腿/腰 peak 略大，但发生时序晚，只有相关证据。

### 3. 共同的闭环异常顺序是 history/姿态支持域 → tilt → 动作/关节 → collapse

相对较宽的 Stage335 90-state reference，三组中位事件为：

| 事件 | source | f000 | f005 |
|---|---:|---:|---:|
| previous-action OOD | +0.26 s | +0.28 s | +0.24 s |
| projected-gravity OOD | +1.32 s | +1.34 s | +1.10 s |
| root tilt >0.30 | +1.46 s | +1.48 s | +1.22 s |
| base angular velocity OOD | +1.74 s | +1.76 s | +1.46 s |
| root z <0.45 | +2.06 s | +2.10 s | +1.80 s |

Stage335 的 q/dq group OOD 更晚：source/f000 通常在 `~2.0–2.2 s`，f005 在 `~1.7–1.9 s`，接近或晚于 collapse。因而“q/dq 先爆出训练域，随后机器人倒下”不成立；更一致的顺序仍是 previous-action 组合结构先离域，gravity/tilt 随后恶化。

recorder-v2 exact actor-source reference 更窄：previous-action、projected gravity 和部分 q group 在 handoff 即 OOD。这证明当前 handoff 与已采 v2 suffix 支持不等价，但不能直接定因，因为 v2 reference 只有 Phase19 四条 source episode，覆盖范围有限。相对 Stage335 与相对 v2 的结果必须并列，不能挑一个写成真因。

### 4. f005 没有改变失败顺序，反而更早坍塌

| 物理首次越界中位数 | source | f000 | f005 |
|---|---:|---:|---:|
| tilt >0.30 | +1.46 s | +1.48 s | **+1.22 s** |
| root z <0.45 | +2.06 s | +2.10 s | **+1.80 s** |

f005 比 source 提前约 `0.24–0.26 s` 进入 tilt/height failure。它没有把 common sequence 变成 recovery，只使倒下后停止漂移更快。因此 Phase21 的 stop drift/settle 改善不能被解释为机器人恢复得更好。

### 5. 支撑与上肢边界

- 15/15 在 tilt 越界时 contact generator 都显示 double support；gait/contact-generator group 没有越 LOO-p95。
- 这只能说明 actor 内部相位没有切换异常，不能证明两只物理脚真的承重或足底载荷合理；trace 没有真实/仿真 foot contact wrench、GRF、COP。
- upper target 基本固定，excursion 约 `4.8e-8 rad`；不支持“上肢扰动导致共同失稳”。
- PD tracking error 不是 15/15 的共同先行事件，故不能凭本报告把 PD 写成根因。

## 结论

当前能诚实支持的是一个时序链，而不是因果定论：

```text
move阶段 heading/lateral 已越界
        ↓
recovery handoff时 previous-action / v2姿态支持不匹配
        ↓
Stage335 previous-action组合约+0.24~0.28s离域
        ↓
gravity/tilt约+1.1~1.5s恶化
        ↓
current action、q/dq随后离域
        ↓
root height约+1.8~2.1s坍塌
```

Phase13 已排除首 tick jump；Phase21 又表明给这些 state role 做 5-update recovery 训练没有修复闭环。证据更指向“固定交权发生在 actor-history/姿态不受当前 recovery 数据支持的状态”，但相关不等于因果。

## 唯一最小可证伪下一干预

不训练、不换模型、不改 PD：

```text
control:
固定 stop +2.0s 交给 recovery

candidate:
+2.0s 后继续现有 brake，直到同时满足：
1. contact generator = double support
2. previous-action 落回冻结 recorder-v2 LOO-p95 支持域
然后仍用同一个 recovery actor、同一个0.5s blend交权
```

若延迟到 in-support handoff 后 tilt/root-z 顺序和坍塌时间仍不变，则“history/support timing 是主因”被证伪，应转查 brake/centroidal/contact physics；若坍塌显著推迟或消失，才支持状态门控交权。这是一个 event-policy 单变量实验，不解锁训练或参数扫描。

## 边界

- 纯仿真、纯离线；没有新的 official episode；
- actor generator contact 不是物理接触真值；
- v2 reference 覆盖有限，OOD 是支持域诊断而非 failure oracle；
- 没有训练、真机、WBT、Git 或百度网盘操作。
