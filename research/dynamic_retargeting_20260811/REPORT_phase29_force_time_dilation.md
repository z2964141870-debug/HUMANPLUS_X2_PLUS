# Phase29 — X2 centroidal-force 时间放慢 A/B

## 裁决

`TIME DILATION IMPROVES BUT DOES NOT SOLVE FORCE FEASIBILITY / ROUTE STOPPED`

唯一变量是冻结 Phase27 路径的时间尺度 `1× → 2×`，即导数步长 `0.02 → 0.04 s`。构型、帧数、接触点、接触带、摩擦系数和 LP 完全相同；未扫描第二个时间尺度。

| 指标 | 1× control | 2× candidate |
|---|---:|---:|
| geometry-qualified | 92/98 | 92/98 |
| force-LP feasible | 22/98 | 33/98 |
| jointly feasible | 22/98 | 33/98 |
| COM accel p95/max (m/s²) | 3.027 / 3.191 | 0.757 / 0.798 |
| Hdot p95/max (N·m) | 23.879 / 25.498 | 5.970 / 6.375 |

COM acceleration 与 Hdot 按预期降到约四分之一，force-feasible 增加 11 帧，证明执行速度确实是一个贡献因素；但 `33/98` 距离预注册全帧门仍很远，不能靠继续放慢或后验挑帧成为 teacher。

结论：停止 time-dilation 路线。下一生成器必须联合调整构型、centroidal angular momentum 和接触力/接触点；不运行 raw physics 或 RL。本结果仍是官方模型可行性 oracle，不是实机 GRF/COP 真值。

## 产物

- `phase29_force_time_dilation_contract.json`
- `run_phase29_force_time_dilation.py`
- `phase29_result.json`
- `tests/test_phase29_force_time_dilation.py`
