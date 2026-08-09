# Toe→Forefoot Temporal Resolution A/B

## 裁决

- candidate gate: **FAIL**。
- joint step max：`0.18503 → 0.18486 rad/frame`；current-v4 110% 上限为 `0.16840`。
- joint step p95-max：`0.17610 → 0.17761`；上限为 `0.14711`。
- 该实验只改变 IK 内部采样密度；输出仍为 30 Hz、smooth9、full-root、GMR root-z。

| variant | semantic p95 | step max/p95-max | valid contact | L/R slip p95 | L/R swing p50 |
| --- | ---: | --- | ---: | --- | --- |
| `toe_to_forefoot_30hz` | 0.10848 | 0.18503/0.17610 | 9/9 | 0.4077596439502492/0.2280554234539653 | 0.07754000991565145/0.08702315897630379 |
| `toe_to_forefoot_60_to_30` | 0.26721 | 0.18486/0.17761 | 9/9 | 0.8061122962297504/0.3847095416823256 | 0.10205056839838958/0.06568141130428651 |

## 门禁

- semantic_nonregression_vs_30hz: `False`
- joint_step_max_within_current_v4_110pct: `False`
- joint_step_p95_within_current_v4_110pct: `False`
- contact_coverage_7_of_9: `True`
- stance_slip_nonregression_vs_current: `False`
- swing_clearance_not_compressed_vs_current: `True`
- winner: `False`

接触、滑移和 clearance 仍是模型几何估计，不是真实足底力/COP；没有训练。
