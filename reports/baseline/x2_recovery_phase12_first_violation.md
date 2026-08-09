# BASE Phase12：交权后首次越界离线审计

日期：2026-08-09
状态：完成；纯离线，不训练、不运行官方仿真、不修改 adapter。

## 假设

Phase11 的失败可能不是 recovery actor 逐步积累出的缓慢退化，而是固定在 stop 第 2.0 s 的 moving→recovery 交权产生大幅动作不连续，随后经过约 1.5–2.0 s 的物理响应才坍塌。

## 干预

没有控制干预。本阶段只对 Phase11 的 10 条官方 trace 按 stop/handoff 对齐，并审计：

- 交权前 1.98 s 与交权后 2.00 s 的 15D action 差；
- action 差乘以 `LOWER_SCALE` 后的目标关节角差；
- 左腿、右腿、腰部和逐关节贡献；
- 交权后 root height、tilt、水平速度第一次越过现有报告阈值的时间。

这里的“raw”指 trace 已记录的、最终发给执行层的 15D 归一化 combined action，不是 ONNX 未裁剪 mean；目标角差为 `action_delta * LOWER_SCALE`，default pose 在相减中抵消。

## 对照

- source recovery：5 条；
- f005 recovery：5 条；
- 10/10 条的 policy-slot 计数均为 `main=360, stationary=100, recovery=300`；
- 10/10 条均恰好在 stop 2.00 s 进入 recovery，前一帧为 1.98 s。

阈值沿用已有 stop 报告：root z 0.45 m、tilt 0.30 rad。`0.03 m/s` 原本是末尾一秒平均速度门，本报告仅把其用于瞬时速度参考，不把它解释为跌倒原因。

## 结果

### 1. 交权点存在一致的大动作跳变

| 指标（5 条中位数） | source recovery | f005 recovery |
|---|---:|---:|
| normalized action delta L2 | 1.344 | 1.390 |
| normalized action delta L∞ | 0.699 | 0.767 |
| target delta L2 | 0.434 rad | 0.471 rad |
| target delta L∞ | 0.259 rad | 0.307 rad |
| 相对交权前 1 s 相邻帧 target-step 中位数 | 10.72× | 10.02× |

这不是某一条随机 trace 的尖峰：两组所有样本都在同一 2.00 s 边界出现，且 target L2 跳变范围分别为 0.415–0.463 rad 和 0.449–0.484 rad。

### 2. 跳变是全身且明显左右不对称的

| target delta L2 中位数 | source recovery | f005 recovery |
|---|---:|---:|
| 左腿 | 0.333 rad | 0.366 rad |
| 右腿 | 0.211 rad | 0.258 rad |
| 腰部 | 0.179 rad | 0.157 rad |
| 左/右腿 L2 比 | 1.58× | 1.38× |

source 主要贡献为：左膝 `+0.259 rad`、左髋 pitch `-0.188 rad`、右髋 pitch `+0.185 rad`、腰 yaw `+0.172 rad`。f005 主要贡献为：左膝 `+0.307 rad`、右髋 pitch `+0.221 rad`、左髋 pitch `-0.154 rad`、腰 yaw `+0.145 rad`、右髋 roll `-0.118 rad`。

所以它不是单个踝关节或单侧噪声，而是 moving actor/phase 语义切换到 recovery actor 后的腿腰姿态重排。

### 3. 交权时机器人尚未倒，坍塌存在明确延迟

| 指标（中位数） | source recovery | f005 recovery |
|---|---:|---:|
| 交权时 root z | 0.621 m | 0.624 m |
| 交权时 tilt | 0.084 rad | 0.083 rad |
| tilt 首次 >0.30 rad | +1.32 s | +1.60 s |
| root z 首次 <0.45 m | +1.92 s | +2.14 s |

source 的 tilt 越界范围为交权后 1.28–1.52 s、height 越界为 1.88–2.10 s；f005 分别为 1.42–1.64 s 和 1.98–2.20 s。f005 把坍塌延迟约 0.2–0.3 s，却没有阻止它，这与 Phase11“stop drift 略好但仍 0/5”的结果一致。

水平速度超过 0.03 m/s 的瞬时参考通常发生在交权当帧或下一帧；由于该阈值本来只用于末尾一秒均值，它不能证明速度先导致了坍塌。

## 结论

证据支持两个事实：

1. 固定 2.0 s 交权存在可复现的全腿腰、左右不对称动作不连续；
2. 机器人在交权时仍直立，约 1.3–1.6 s 后倾斜越界、1.9–2.2 s 后高度坍塌。

这形成了很强的时序相关证据，但仍不是因果证明：同一批 trace 不能回答“消除跳变是否一定不倒”。因此不能把本报告写成已解决根因，也不支持继续增加 recovery 训练轮数。

## 下一步

若继续，唯一优先的可证伪实验应是同一 source/candidate 合同下的**有界 handoff blend/slew 单变量 A/B**：保持 moving、recovery、交权时刻和所有物理参数不变，只限制交权后的每 tick target delta，并检查动作跳变、tilt/root-z 首次越界是否同步推迟或消失。在此之前，25-update 与长训保持锁定。

可复现工具：[audit_phase12_first_violation.py](../../tools/official_x2/audit_phase12_first_violation.py)
完整逐条结果：[x2_recovery_phase12_first_violation.json](./x2_recovery_phase12_first_violation.json)
