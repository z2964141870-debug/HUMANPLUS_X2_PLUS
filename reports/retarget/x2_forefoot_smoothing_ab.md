# Toe→Forefoot Output Smoothing A/B

## 裁决

- 旧 Task3 timebase 不匹配动作数：`1/9`；最大帧数比例 `1.1083`。
- first passing window: `None`。
- IK、root 与 contact contract 固定；唯一变量是已有对称 moving-average 的窗口。

| window | semantic p95 | step max/p95-max | valid | L/R slip p95 | L/R clearance p50 | winner |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 9 | 0.10928 | 0.18503/0.17610 | 9/9 | 0.3991/0.2851 | 0.0730/0.0913 | FAIL |
| 11 | 0.10978 | 0.15217/0.14373 | 9/9 | 1.1310/0.2530 | 0.0692/0.0900 | FAIL |
| 13 | 0.11387 | 0.13037/0.12036 | 9/9 | 1.5197/0.9059 | 0.0785/0.0723 | FAIL |
| 15 | 0.13682 | 0.11309/0.10500 | 9/9 | 0.9067/1.3093 | 0.0676/0.0690 | FAIL |

这仍是 Bronze 运动学筛选；接触量是模型估计值，必须再过物理四域才可称 Silver。
