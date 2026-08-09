# X2 WBT Generation Contract Phase5

- 裁决：**PHASE5_GENERATION_CONTRACT_NOT_PROMOTABLE**。
- 无训练/teacher搜索/checkpoint/真机；只比较 original、reset-compatible(A)、A+ground(B)。

## 预注册门

A：0.5s静止前缀、1s速度连续time-warp；frame0 joint/root速度近零，连接处joint/root速度跳变受限。B：official signed geometry+intended support的平滑root-z轨迹，|z|≤0.10m、每帧≤4mm。A+B相对A任一动作生存下降>0.1s或slip超过`max(1.1×A, A+0.02)`即停止。

## 离线连续性与ground

| role | A continuity | join q/root-ang jump | z range/step(m) | contact |dist| p95 before→after(m) | deep penetration before→after |
|---|---:|---:|---:|---:|---:|
| walk_straight | True | 0.088/0.011 | -0.023..-0.005/0.0022 | 0.028→0.012 | 0.000→0.000 |
| turn_left | False | 0.377/1.227 | 0.015..0.049/0.0040 | 0.067→0.033 | 0.588→0.152 |
| turn_right | False | 0.241/0.523 | 0.009..0.041/0.0040 | 0.060→0.029 | 0.548→0.095 |
| stand_to_walk | True | 0.134/0.084 | 0.005..0.051/0.0040 | 0.062→0.030 | 0.761→0.075 |
| walk_to_stand | True | 0.284/0.211 | 0.012..0.041/0.0040 | 0.060→0.032 | 0.408→0.085 |

## prescribed/free zero-update paired gate

| role | variant | prescribed full | free survival(s) | slip p95 | contact agreement vs intended | sat. |
|---|---|---:|---:|---:|---:|---:|
| walk_straight | original | 1.000 | 1.557 | 0.0101 | 0.219 | 0.0004 |
| walk_straight | reset_compatible | 1.000 | 1.808 | 0.0081 | 0.209 | 0.0000 |
| walk_straight | reset_plus_ground | 1.000 | 1.751 | 0.0082 | 0.372 | 0.0000 |
| turn_left | original | 1.000 | 1.890 | 0.1490 | 0.675 | 0.0108 |
| turn_left | reset_compatible | 1.000 | 1.573 | 0.0246 | 0.673 | 0.0129 |
| turn_left | reset_plus_ground | 1.000 | 1.589 | 0.0224 | 0.559 | 0.0009 |
| turn_right | original | 1.000 | 1.117 | 0.0987 | 0.677 | 0.0123 |
| turn_right | reset_compatible | 1.000 | 1.134 | 0.0076 | 0.661 | 0.0141 |
| turn_right | reset_plus_ground | 1.000 | 1.148 | 0.0076 | 0.551 | 0.0010 |
| stand_to_walk | original | 1.000 | 1.603 | 0.0059 | 0.790 | 0.0055 |
| stand_to_walk | reset_compatible | 1.000 | 1.605 | 0.0030 | 0.807 | 0.0052 |
| stand_to_walk | reset_plus_ground | 1.000 | 1.630 | 0.0030 | 0.530 | 0.0009 |
| walk_to_stand | original | 1.000 | 1.952 | 0.0671 | 0.635 | 0.0109 |
| walk_to_stand | reset_compatible | 1.000 | 2.665 | 0.1419 | 0.652 | 0.0112 |
| walk_to_stand | reset_plus_ground | 1.000 | 1.839 | 0.0627 | 0.495 | 0.0010 |

## 结论

- 结果：至少一项预注册连续性、ground或官方物理门失败。
- 结论：只否定当前最小生成器修复；禁止turn teacher/PPO，并保留original作为对照。
- 下一步：定位失败门，不扩大参数或搜索。

若失败，结论只否定当前reset/ground生成器，不构成X2动力学不可能证明。
