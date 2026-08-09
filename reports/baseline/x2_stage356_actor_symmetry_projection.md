# X2 Stage356：Actor 左右对称投影单变量裁决

日期：2026-08-09

训练更新：`0`

真机：未使用
裁决：`ROLL_YAW_SYMMETRY_PROJECTION_REJECTED`

## 假设

Stage351 在直行命令下产生 `+0.426 rad` 偏航和 `+0.385 m` 横漂，而完整镜像 Stage355 将符号翻成 `-0.615 rad`、`-0.590 m`。这说明冻结 actor 的左右不等变性是 stiff 域方向漂移的重要机制。若只把 roll/yaw 动作投影到 actor 与镜像 actor 的群平均，横漂和偏航应同时小于两个对照，并且不得破坏存活。

## 干预

保持 Stage351 的 checkpoint、PD、命令、起步/行走/减速/停止时序、stand/recovery actor 和官方 AimDK v1.0 MuJoCo 全部不变。仅在 moving main actor 上执行：

```text
a_eq = 0.5 * (a(obs) + M^-1 a(M obs))
```

只覆盖 lower-15 中名字包含 `roll` 或 `yaw` 的动作；pitch、knee、upper、stand/recovery 路径不改。`alpha=1.0`，没有参数扫描。

## 对照与结果

| 条件 | forward | lateral | yaw progress | max heading | stop min root-z | 是否倒地 |
|---|---:|---:|---:|---:|---:|---:|
| Stage351 原 actor | `+1.358 m` | `+0.385 m` | `+0.426 rad` | `0.438 rad` | `0.619 m` | 否 |
| Stage355 完整镜像 | `+1.375 m` | `-0.590 m` | `-0.615 rad` | `0.625 rad` | `0.620 m` | 否 |
| Stage356 roll/yaw 群平均 | `-0.087 m` | `-0.151 m` | `-0.538 rad` | `1.303 rad` | `0.134 m` | **是** |

Stage356 的 startup 门通过，但行走随后反向、航向误差扩大并倒地；move/stop/full gate 均失败。340 个 moving inference 的诊断为：

- actor equivariance RMSE mean：`1.07498`；
- actor equivariance abs max：`8.67389`；
- 实际投影 action delta RMSE mean：`0.49038`；
- 实际投影 action delta abs max：`4.33694`（之后仍按既有部署契约 clip）。

## 结论

左右不等变性确实存在，但“部署时直接把 roll/yaw 动作做群平均”不是安全修复。它改变了 actor 已学到的跨关节闭环配合，即使 pitch/knee 没有直接被投影，也足以把稳定前行破坏成反向失稳。按预注册停止条件，本分支立即停止；不减 alpha、不换 mask、不做参数扫描。

本实验也不处理 Stage250 约 `10–11°` 的持续后仰。后仰是独立的矢状面姿态缺陷，不能用这次左右对称失败掩盖。

## 下一步

1. 保留左右不等变性作为训练诊断指标，而不是部署时 action hack。
2. 若未来重训 BASE actor，应在训练期加入与原 gait template/phase 一致的 symmetry loss，并用官方域验证；不能对冻结输出强行平均。
3. 当前更优先修正 signed pitch/默认姿态与停止后端；Stage250 只保留为基本位移能力基线。
4. WBT 继续先修 reset/ground/contact 生成契约，不因本 BASE 负结果改变路线。

机器可读证据见 `x2_stage356_actor_symmetry_projection.json`；原始 official trace 位于现有 AimDK 结果目录。
