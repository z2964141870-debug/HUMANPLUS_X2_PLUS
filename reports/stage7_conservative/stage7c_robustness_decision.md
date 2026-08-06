# Stage7C 成对扰动鲁棒性裁决

结论：**KEEP_STAGE208_BASE**。

| 分支 | 生存 | 总步数 | 低速步数 | 世界 heading | 相对 heading | lateral | tilt |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 15/36 | 9674 | 2788 | 0.5988 | 0.5941 | 0.4657 | 0.5465 |
| NEUTRAL05 | 17/36 | 9961 | 2841 | 0.5697 | 0.5651 | 0.4872 | 0.5111 |

## 分 seed

| seed | BASE steps | NEUTRAL steps | BASE heading | NEUTRAL heading | BASE lateral | NEUTRAL lateral | 三项不劣 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 17 | 3251 | 3373 | 0.6468 | 0.6187 | 0.4741 | 0.5116 | ✗ |
| 42 | 3272 | 3214 | 0.5498 | 0.5469 | 0.4671 | 0.4468 | ✗ |
| 73 | 3151 | 3374 | 0.5997 | 0.5434 | 0.4560 | 0.5031 | ✗ |

## 预注册门

- ✓ `initial_conditions_exactly_paired`
- ✓ `survival_not_below_base`
- ✓ `steps_not_below_base`
- ✓ `low_speed_steps_not_below_base`
- ✓ `world_heading_not_above_base`
- ✗ `lateral_not_above_base`
- ✗ `base_normal_survivors_preserved`
- ✗ `no_normal_heldout_regression`
- ✗ `at_least_two_of_three_seeds_non_worse`
- ✓ `residual_within_0p05`

完全配对：`True`；满足三项不劣的 seed：`0/3`。
