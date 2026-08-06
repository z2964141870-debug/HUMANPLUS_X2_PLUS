# Stage7 residual blend 预注册裁决

结论：**STATIC_BLEND_FAILED_HELDOUT**。

仅按 selection 锁定的 alpha：`0.75`。

| alpha | selection通过 | heading | lateral | heading改善 | lateral改善 | 归一化分数↓ |
|---:|:---:|---:|---:|---:|---:|---:|
| 0.25 | ✓ | 0.6015 | 0.9169 | 18.2% | 12.3% | 1.6947 |
| 0.50 | ✓ | 0.4893 | 0.7452 | 33.4% | 28.8% | 1.3780 |
| 0.75 | ✓ | 0.4161 | 0.5090 | 43.4% | 51.3% | 1.0525 |

## 留出门

- ✓ `survival_count_not_below_base`
- ✓ `all_base_normal_speed_survivors_preserved`
- ✗ `no_normal_speed_heldout_regression`
- ✓ `female_lift_steps_not_below_base`
- ✗ `low_speed_steps_not_below_base`
- ✓ `full_panel_mean_heading_better_than_base`
- ✓ `full_panel_mean_lateral_better_than_base`

## 留出回归

- `wave_left_heldout__vx0.30`：heading_worse_than_base_plus_0p15, surviving_case_tilt_gt_0p45

Phase B 风险约束训练解锁：`True`。
