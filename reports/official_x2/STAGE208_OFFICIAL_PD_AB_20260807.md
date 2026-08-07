# Stage208 × 官方 X2 MuJoCo：PD 契约 A/B（2026-08-07）

## 结论

Stage208 在官方域开局倒地的主导控制差异已收窄到**踝关节刚度**：只把左右踝 Kp 从训练侧的 300 改为官方样例使用的 40，机器人首次在官方 MuJoCo 中连续运行完整 8 秒而不倒。该修正解决了生存问题，但未解决航向和停止，因而是有效地基修复，不是最终 locomotion 通过。

## 假设

Stage208 在 IsaacLab 中使用的统一腿腰 `Kp/Kd=300/20` 与官方 X2 分组控制契约相差过大，可能导致相同 position target 在官方 MuJoCo 中产生不同闭环。

## 干预与对照

保持 Stage208 ONNX、默认姿态、action scale、gait template、50 Hz、官方 reset、官方 MJCF 和命令时序不变，只替换腿腰 PD：

| 分支 | Kp | Kd |
|---|---|---|
| Stage208 control | 全腿腰 300 | 全腿腰 20 |
| official native | 官方逐关节 | 官方逐关节 |
| official Kp only | 官方逐关节 | 20 |
| official Kd only | 300 | 官方逐关节 |
| proximal only | 仅髋膝使用官方 Kp | 20 |
| ankle only | 仅踝使用官方 Kp=40 | 20 |
| waist only | 仅腰使用官方 Kp | 20 |

## 结果

| 分支 | 高度首次失败 | 倾角首次失败 | 1 s 倾角 | 8 s root z min | 结论 |
|---|---:|---:|---:|---:|---|
| Stage208 control | 1.50 s | 1.32 s | 0.495 rad | 0.167 m | 失败 |
| official native | 2.28 s | 1.98 s | 0.387 rad | 0.089 m | 延后但失败 |
| official Kp only | 2.44 s | 2.50 s | 0.252 rad | 0.104 m | 延后但失败 |
| official Kd only | 1.58 s | 1.40 s | 0.423 rad | 0.155 m | 改善很小 |
| proximal only | 4.24 s | 3.90 s | 0.323 rad | 0.120 m | 明显延后但失败 |
| **ankle only** | **无** | **无** | **0.259 rad** | **0.652 m** | **8 s 生存通过** |
| waist only | 1.50 s | 1.34 s | 0.484 rad | 0.134 m | 基本无效 |

`ankle only` 在重复 0.30 m/s 和 0.20 m/s 测试中继续保持高度与倾角稳定，说明不是单次偶然结果。

## 新暴露的失败模式

稳定不等于正确行走：

- 0.30 m/s 重复测试在 8 秒内约 `Δx=+0.83 m, Δy=-1.42 m`。
- root yaw 累计 `-1.16 rad`，说明主要是持续右转，不只是世界系横滑。
- body-frame 平均速度约为前向 `0.203 m/s`、横向 `-0.079 m/s`。
- 切换到零速度命令后的 2 秒仍漂移约 `0.51 m`，并继续偏航约 `-0.52 rad`。

因此当前严格裁决为：

1. 官方域开局失稳的踝刚度硬冲突已解决；
2. Stage208 能在官方域形成稳定但错误的弯行轨迹；
3. 航向和停止命令尚未闭环，不能部署，也不能宣称 locomotion 通过。

## 下一步

在 `ankle Kp=40` 的稳定基底上做 `actor_only / template_only / full` 三分解，并保留 yaw、body-frame 速度和停止段指标。该实验直接判断持续右转来自 gait template 的非对称性，还是 Stage208 actor 在官方域的闭环偏置。
