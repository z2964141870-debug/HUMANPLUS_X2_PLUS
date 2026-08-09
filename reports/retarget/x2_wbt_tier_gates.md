# X2 WBT Bronze / Silver / Gold 自动审计门禁草案

版本：Phase0-v1
状态：预注册草案；不授权 PPO 或批量重定向。

## 核心原则

这三层回答三个不同问题：

- Bronze：动作在官方 X2 几何中是否是连续、有限、语义正确的运动学结果；
- Silver：动态动作的足底接触、摆动脚和 root 是否彼此一致；
- Gold：同一轨迹在 AimDK v1.0 官方 MuJoCo 中是否真的可由预注册控制器执行。

Silver 不是“放松版 Gold”，Bronze 静态动作也不会因为双脚不动自动成为动态 Silver。FK/contact/COM 均必须注明是官方模型估计值，不能写成实测足底力或 COP。

## 全局 Reject

任何一项成立即拒绝：NaN/Inf、joint order 不符、头未按 29DoF contract 锁定、硬限位超出超过 `1e-6 rad`、source hash 缺失、root quaternion 误差超过 `1e-5`、root XY 修补超过 `0.20 m`，或通过固定 root-z、删除难段、强制双脚接触、米级 root 修补、事后 clamp 来伪造结果。

## Bronze：运动学可用

| 指标 | 门槛 | 方向 |
| --- | ---: | --- |
| joint step p95 / max（30 Hz） | `≤0.10 / 0.15 rad` | 越小越好 |
| sole penetration p95 / max | `≤5 / 15 mm` | 越小越好 |
| tracked keypoint error p95 / max | `≤0.10 / 0.25 m` | 越小越好 |
| pelvis height | `[0.40, 1.00] m` | 区间内 |
| root tilt | `≤0.80 rad` | 越小越好 |
| mirror joint round-trip | `≤1e-12 rad` | 越小越好 |
| mirror FK position / rotation-element | `≤3 mm / 0.007` | 越小越好 |

还必须人工/规则核对动作方向、左右语义、动作类别以及明显自碰撞。3 mm 镜像 FK 容差来自官方 CAD 已验证的毫米级左右非对称，不是放宽关节映射。

## Silver：接触可用

仅对动态动作定义，且必须先过 Bronze：

| 指标 | 门槛 |
| --- | ---: |
| 完整 DS→SS→DS 周期 | `≥1` |
| 每个意图脚的 contact transitions | `≥2` |
| 每脚 stance speed p95 | `≤0.10 m/s` |
| 每次 stance excursion | `≤0.03 m` |
| 每个意图摆动脚 clearance p50 / p95 | `≥10 / 20 mm` |
| contact timing error | `≤0.10 s` |
| 非预期 flight fraction | `≤2%` |
| root horizontal acceleration p95 | `≤4.0 m/s²` |

standing/upper-only 可以作为 Bronze 稳定和上肢保持数据，但不能冒充 locomotion Silver。

## Gold：官方仿真动态可执行

必须在 AimDK v1.0 官方 X2 MuJoCo 中用预注册的 position-PD 或 teacher 完整回放；报告必须冻结默认姿态、逐关节增益、action scale 和频率。

| 指标 | 门槛 |
| --- | ---: |
| full-clip survival | `100%` |
| root position RMSE / p95 | `≤0.10 / 0.20 m` |
| root orientation mean / p95 | `≤10° / 20°` |
| body MPJPE mean / p95 | `≤0.10 / 0.15 m` |
| stance slip p95 | `≤0.15 m/s` |
| contact timing error | `≤0.12 s` |
| single-support ratio / reference | `≥70%` |
| joint target step p95 / max | `≤0.10 / 0.20 rad` |
| action saturation | `≤5%` |
| effort-limit violation | `0` |

踢腿/抬腿必须真实卸载摆动脚；弓步必须保留指定前脚、步距、pelvis 高度和躯干方向；深蹲必须保留 root-z 变化；转向必须保留世界 yaw 方向和幅值。

## 一次性校准规则

这些绝对值先用于 Phase0 诊断。全量扫描前只允许基于固定 24 条诊断面板做一次公开修订，并给出阈值 old/new diff；held-out 结果禁止用于调门。机器可读定义见 `x2_wbt_tier_gates.json`。
