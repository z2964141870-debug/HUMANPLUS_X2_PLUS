# Stage7B Gate5 裁决

结论：**STOP_RISK_KEEP_NEUTRAL_FOR_ROBUSTNESS**。

| 分支 | 生存 | 总步数 | 低速步数 | heading | lateral | tilt |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 5/12 | 3176 | 962 | 0.6567 | 0.5259 | 0.5384 |
| NEUTRAL05 | 5/12 | 3248 | 974 | 0.6031 | 0.4996 | 0.5544 |
| RISK05 | 5/12 | 3175 | 966 | 0.6482 | 0.5072 | 0.5446 |

## RISK05 预注册门

- ✓ `survival_not_below_neutral`
- ✗ `steps_not_below_neutral`
- ✗ `heading_not_above_neutral`
- ✗ `lateral_not_above_neutral`
- ✓ `base_normal_survivors_preserved`
- ✓ `wave_left_heading_and_tilt_safe`
- ✓ `heldout_heading_or_survival_improved_vs_neutral`

## NEUTRAL05 意外 Pareto 候选审计

- ✓ `survival_not_below_base`
- ✓ `steps_not_below_base`
- ✓ `low_speed_steps_not_below_base`
- ✓ `heading_better_than_base`
- ✓ `lateral_better_than_base`
- ✓ `base_normal_survivors_preserved`
- ✓ `no_normal_heldout_regression`
- ✓ `residual_within_0p05`

NEUTRAL05 robustness 复测候选：`True`。
