# Stage6 Gate5 物理裁决

结论：**UNLOCK_GATE25**。

> 该门只决定是否允许做 25-update 验证，不代表 Stage6 已成功。

| 分支 | 生存 | 总步数 | 平均航向峰值 rad | 平均横漂 m | 平均倾角峰值 rad | residual 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 5/12 | 3176 | 0.6567 | 0.5259 | 0.5384 | 0.0000 |
| CURRENT | 5/12 | 3171 | 0.6428 | 0.5203 | 0.5417 | 0.0427 |
| FUTURE | 5/12 | 3170 | 0.6141 | 0.4735 | 0.5446 | 0.0322 |
| FUTURE-NOPHASE | 4/12 | 3115 | 0.6570 | 0.5114 | 0.5816 | 0.0316 |

## 预注册门

- ✓ `future_survival_not_below_base`
- ✓ `failed_or_low_speed_case_improved`
- ✓ `future_heading_better_than_current`
- ✓ `future_heading_better_than_future_nophase`
- ✓ `future_lateral_better_than_current`
- ✓ `future_lateral_better_than_future_nophase`
- ✓ `normal_speed_regression_panel_passed`
- ✓ `future_residual_within_0p10`

## 低速共同窗口改善

- `wave_real__vx0.20`：共同窗口 167/169 步，航向改善 0.1116 rad，横漂改善 0.0813 m。

低速改善只在候选轨迹不少于 BASE 两个控制帧、且航向和横漂都至少改善 0.01 时计数，避免更早终止造成的假低峰值。
