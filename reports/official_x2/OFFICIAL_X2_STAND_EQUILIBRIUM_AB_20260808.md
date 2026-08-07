# 官方 X2 MuJoCo 站立平衡中心 A/B（2026-08-08）

## 一句话结论

Stage208 的停止失败不是 gait-phase 编码错误，也不是换成官方默认蹲姿就能修复：纯 PD 保持两种姿态都会倒，Stage208 policy 能保住高度却会在零命令下持续行走，因此当前缺失的是真正的 stand/stop 闭环吸引域。

## 假设

Stage208 在 IsaacLab 的默认下肢姿态为髋/膝/踝 `-0.248 / 0.530 / -0.282 rad`，官方 v1.0 RL deploy 默认姿态为 `-0.312 / 0.669 / -0.363 rad`。官方模型的物理平衡中心可能更接近后者。

## 干预

- 固定 Stage208-s2550 ONNX、官方 MuJoCo、50 Hz、`official_kp_ankle` PD 与 8 s 时窗。
- 只切换 `stage208` / `official_v1` 默认姿态。
- 对每种姿态分别评估：零命令 Stage208 policy；绕开 policy 的纯 PD 保持。
- 静止 gait observation 已对照训练源码，两端均为 `[0, 0, 1, 1]`。

## 结果

| 控制 | 默认姿态 | 8 s XY 漂移 | z min/final | 最大倾角 | 末 1 s 平均速度 | 结果 |
|---|---|---:|---:|---:|---:|---|
| Stage208 policy | Stage208 | 1.412 m | 0.586/0.647 m | 0.418 rad | 0.150 m/s | 不倒，但零命令下行走 |
| Stage208 policy | official v1 | 2.572 m | 0.590/0.614 m | 0.200 rad | 0.414 m/s | 倾角较小，漂移明显恶化 |
| 纯 PD | Stage208 | 0.789 m | 0.105/0.121 m | 1.618 rad | 0.000 m/s | 倒地 |
| 纯 PD | official v1 | 0.754 m | 0.108/0.127 m | 1.661 rad | 0.000 m/s | 倒地 |

## 结论

1. 官方深蹲姿态对正立倾角有局部收益，但不能让零命令 policy 停下，不应接入默认部署。
2. 纯 PD 的失败说明官方 MuJoCo 中必须有动态平衡反馈；不能用“停止时固定默认角”代替 stand policy。
3. Stage208 的失败模式是“用持续踏步维持稳定”，而不是“不会保持高度”。训练中 `rel_standing_envs=0.2` 并未转化成官方域可用的静止技能。

## 下一步

- 不再扫默认姿态或简单 latch 阈值。
- 将 stand 从 locomotion policy 中独立成最小后端技能，首先要求官方 MuJoCo 内 20–60 s 的低漂移稳定，再做 `velocity_brake → stand` 切换。
- 已在评估器中增加显式 `stand_root_*`、末 1 s 速度与 stand 高度门，避免纯 stand 实验被旧的 move-only summary 误报为空值。

## 原始结果

`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807/`

- `stand-policy-stage208-8s.json`
- `stand-policy-official-v1-8s.json`
- `stand-pd-stage208-8s.json`
- `stand-pd-official-v1-8s.json`
