# Toe→Forefoot Bilateral-Knee Continuity A/B

## 裁决

- first passing knee window: `None`。
- 29/31 DoF 完全保留 smooth9；仅左右 knee 使用更宽窗口。

| knee window | semantic p95 | step max/p95-max | valid | L/R slip p95 | L/R clearance p50 | winner |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 11 | 0.10986 | 0.15217/0.14373 | 9/9 | 0.9513/0.3036 | 0.0734/0.0774 | FAIL |
| 13 | 0.10977 | 0.13851/0.12401 | 9/9 | 1.3312/0.4589 | 0.0844/0.0800 | FAIL |
| 15 | 0.12104 | 0.13851/0.12349 | 9/9 | 1.7167/0.4702 | 0.0844/0.0766 | FAIL |

该门仅决定 Bronze 候选；模型接触估计不等于真实 GRF/COP，也不替代物理四域回放。
