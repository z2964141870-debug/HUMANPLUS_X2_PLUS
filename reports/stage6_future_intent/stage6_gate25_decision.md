# Stage6 Gate25 最终裁决

结论：**KEEP_STAGE208**（PARTIAL_SIGNAL）。

FUTURE 的总体方向指标确有信号，但未满足跨动作的一致安全改善；因此 Stage208 继续作为可保留基线，Stage6 仅作为研究候选。

| 分支 | 生存 | 总步数 | 低速总步数 | 平均航向峰值 rad | 平均横漂 m | 平均倾角峰值 rad |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 5/12 | 3176 | 962 | 0.6567 | 0.5259 | 0.5384 |
| CURRENT | 5/12 | 3108 | 913 | 0.6421 | 0.5500 | 0.5465 |
| FUTURE | 5/12 | 3218 | 945 | 0.5597 | 0.2822 | 0.5741 |
| FUTURE-NOPHASE | 5/12 | 3125 | 970 | 0.7017 | 0.5706 | 0.5366 |

## Gate25 一致性门

- ✓ `future_survival_count_not_below_any_control`
- ✓ `future_total_steps_not_below_any_control`
- ✓ `future_mean_heading_better_than_all_controls`
- ✓ `future_mean_lateral_better_than_all_controls`
- ✓ `all_base_surviving_cases_preserved`
- ✗ `low_speed_total_steps_not_below_base`
- ✗ `no_normal_speed_casewise_safety_or_direction_regression`
- ✓ `future_residual_within_0p10`

## 正常速度回归

- `wave_left_heldout__vx0.30`：heading_worse_than_base_plus_0p15, surviving_case_tilt_gt_0p45

## FUTURE 相对 BASE 的逐案例变化

| 案例 | 生存步数 Δ | 航向峰值 Δ rad | 横漂 Δ m | 倾角 Δ rad |
|---|---:|---:|---:|---:|
| box_lift__vx0.20 | -2 | +0.0729 | +0.0216 | -0.0004 |
| box_lift__vx0.30 | +0 | -0.5084 | -0.6779 | +0.0574 |
| female_lift_heldout__vx0.20 | -3 | +0.0119 | -0.0715 | +0.0242 |
| female_lift_heldout__vx0.30 | +59 | +0.0418 | +0.0329 | -0.0083 |
| knocking__vx0.20 | +1 | +0.0410 | -0.0043 | +0.0255 |
| knocking__vx0.30 | +0 | -0.4755 | -0.7614 | +0.0736 |
| swing_arms__vx0.20 | -1 | +0.0163 | -0.0068 | -0.0048 |
| swing_arms__vx0.30 | +0 | -0.4707 | -0.5404 | +0.0253 |
| wave_left_heldout__vx0.20 | -3 | +0.0377 | +0.0137 | -0.0161 |
| wave_left_heldout__vx0.30 | +0 | +0.5580 | -0.4400 | +0.2096 |
| wave_real__vx0.20 | -9 | -0.0432 | -0.0500 | +0.0050 |
| wave_real__vx0.30 | +0 | -0.4457 | -0.4395 | +0.0383 |

Δ 小于 0 对航向、横漂、倾角更好；生存步数 Δ 大于 0 更好。
