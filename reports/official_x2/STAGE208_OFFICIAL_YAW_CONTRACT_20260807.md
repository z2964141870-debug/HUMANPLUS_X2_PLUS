# Stage208/212 × 官方 X2 MuJoCo：航向契约裁决（2026-08-07）

## 结论

在官方踝 Kp=40 的稳定基底上，Stage208 可以连续运行 8 秒，但无法可靠响应 yaw-rate command，也无法在零速度命令后停止。持续右转不是 gait template 单独造成的，也不是 URDF/MJCF yaw 轴符号不一致；它是 Stage208 本身已有的弱且非对称 yaw authority 在官方域被放大的结果。现有 Stage212 yaw specialist 在官方域直接失稳，不能替代 Stage208。

## Actor / template 责任拆分

| 分支 | 8 s 生存 | yaw 累计 | `Δx/Δy` | 结论 |
|---|---:|---:|---:|---|
| full | 是 | -1.161 rad | +0.836/-1.429 m | 稳定但持续右转 |
| actor only | 是 | -0.882 rad | +1.114/-1.452 m | 主要偏航经 actor 闭环产生 |
| template only | 否 | -0.016 rad | -0.754/+0.009 m | 模板不制造偏航，但不能独立维持平衡 |

## Heading-hold 对照

当前官方适配器最初漏掉了 Stage208 训练/旧评估使用的 heading-hold。恢复同构的世界航向误差到 yaw-rate command 后：

| heading gain | 8 s yaw 累计 | `Δx/Δy` | 生存 |
|---:|---:|---:|---:|
| 0.00 | -1.161 rad | +0.836/-1.429 m | 是 |
| 0.05 | -0.922 rad | +1.029/-1.339 m | 是 |
| 0.20 | -1.202 rad | +0.874/-1.343 m | 是 |
| 1.00 | -0.920 rad | +0.958/-1.205 m | 是 |

恢复契约是必要修正，但提高外环增益不能闭合航向。

## 正负 yaw 命令方向辨识

Stage208 在官方域的 `wz=0/+0.2/-0.2` 平均实际 yaw rate 分别约为：

```text
-0.145 / -0.146 / -0.120 rad/s
```

三者都向负 yaw 转动，说明策略在该状态分布中对 yaw command 的控制权远小于零命令偏置。离线固定同一 observation 时，ONNX 的 `wz=+0.2` 与 `wz=-0.2` 输出 L2 差仅 `0.0364`，进一步确认输入灵敏度很弱。

该结果与旧 Stage210/211 IsaacLab 诊断一致：Stage208 在自建 nominal/ideal 域中也曾出现正负命令均偏负 yaw。因此官方仿真揭示的是已有策略缺陷，不是新引入的接口 bug。

## 排除项

- 官方 MJCF 与 Isaac 使用的 sole12 URDF 中，左右 hip yaw 与 waist yaw 轴均为 `0 0 1`。
- gait template 单独运行几乎无累计 yaw。
- heading gain 0.05–1.0 的非单调结果否定“只需提高外环增益”。
- 不采用瞬时左右对称投影；旧 Stage188/205 已证明非对称动作可能是稳定补偿，硬删除会破坏闭环。

## Stage212 specialist 官方域复查

Stage212-s2649 已无损导出：PyTorch/ONNX 最大误差 `4.77e-7`。在相同官方踝 Kp 基底下：

| yaw command | 8 s 生存 | yaw 累计 | root z min |
|---:|---:|---:|---:|
| 0.0 | 否 | -0.895 rad | 0.149 m |
| +0.2 | 否 | -0.956 rad | 0.150 m |
| -0.2 | 否 | -0.911 rad | 0.147 m |

Stage212 在官方域既未保持步态，也未体现可用的双向 yaw authority，故淘汰为官方后端候选。

## 当前最佳与下一步

当前官方域最佳仍为：

```text
Stage208-s2550 + gait template + ankle Kp=40 / Kd=20
```

它通过 8 秒高度/倾角生存，但不通过航向和停止门。下一步不应继续扫 heading gain 或直接换 checkpoint；应在官方 MuJoCo 中，以冻结 Stage208 为基底，只搜索/学习一个**有界、零输入严格回退、作用于 yaw 协调子空间的低维 adapter**，并同时约束生存、yaw、横漂与停止。若最小 teacher 都找不到可行 correction，再回到官方域 matched training，而不是继续对 Isaac 单域结果晋级。

## 低维 yaw adapter 预筛追加结果

官方域单变量辨识确认“左右 hip yaw 共同方向”是有效控制子空间：

| 干预 | 8 s yaw 累计 | `Δx/Δy` | root z min | tilt max |
|---|---:|---:|---:|---:|
| 无 adapter | -1.161 rad | +0.836/-1.429 m | 0.652 m | 0.313 rad |
| hip yaw 常量 +0.10 | -0.718 rad | +1.350/-1.324 m | 0.651 m | 0.317 rad |
| hip yaw feedback, gain 0.50, max 0.30 | -0.625 rad | +1.455/-1.328 m | 0.652 m | 0.311 rad |
| **hip yaw feedback, gain 1.00, max 0.50** | **-0.371 rad** | **+1.627/-1.234 m** | **0.652 m** | **0.312 rad** |

反馈形式为 `bias=clip(gain × wrapped_heading_error, ±max)`，在零航向误差时严格回退 Stage208；只作用于左右 hip yaw 的 normalized action。它在不损坏生存的情况下将偏航缩小约 68%，证明低维纠偏方向有效，但仍未通过 `|yaw|<=0.30 rad`，横漂和停止也尚未通过。下一轮先验证该候选的停止、低速与重复性，不继续无界放大。
