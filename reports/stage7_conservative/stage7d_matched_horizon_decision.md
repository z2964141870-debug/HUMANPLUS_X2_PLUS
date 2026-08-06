# Stage7D 生存时长删失诊断

结论：**STOP_CURRENT_FUTURE_ADAPTER_OBJECTIVE**。

| 指标（36 对共同窗口） | BASE | NEUTRAL05 | 差值 |
|---|---:|---:|---:|
| lateral max 的 case 均值 (m) | 0.458858 | 0.456815 | -0.002043 |
| lateral 时间均值的 case 均值 (m) | 0.178080 | 0.180017 | +0.001937 |
| 世界 heading 时间均值 (rad) | 0.239104 | 0.237962 | -0.001142 |

## 预注册诊断门

- ✓ `matched_lateral_max_not_above_base`
- ✗ `matched_lateral_time_mean_not_above_base`
- ✓ `matched_world_heading_time_mean_not_above_base`
- ✗ `no_common_survivor_lateral_plus_0p15`
- ✓ `rescued_pre_failure_has_no_lateral_plus_0p15`
- ✓ `trace_rerun_exactly_reproduces_stage7c`

## BASE 失败、候选生存

| seed | case | 共同步数 | 共同窗口 lateral max 差值 (m) |
|---:|---|---:|---:|
| 17 | female_lift_heldout__vx0.30 | 340 | +0.1075 |
| 17 | wave_real__vx0.30 | 341 | -0.0162 |
| 73 | female_lift_heldout__vx0.30 | 205 | +0.0171 |

## BASE 生存、候选失败

| seed | case | 共同步数 | 共同窗口 lateral max 差值 (m) |
|---:|---|---:|---:|
| 42 | female_lift_heldout__vx0.30 | 320 | +0.0686 |

删失偏差得到部分确认：共同窗口 lateral max 略好，且三个被救活案例在 BASE 失效前均未超过 `+0.15 m`；但共同窗口 lateral 时间均值仍轻微恶化，另有一个双方生存的 wave 案例超过门槛。因此不能把候选晋级，也不解锁沿当前目标继续训练。
