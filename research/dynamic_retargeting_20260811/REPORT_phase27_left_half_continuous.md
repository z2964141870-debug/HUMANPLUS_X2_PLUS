# Phase27 — X2 左半周期连续加载/接触路径

## 裁决

`LEFT HALF CONTINUOUS GEOMETRY PASSED / CENTROIDAL FORCE NEXT`

只使用 Phase26 已严格可行的前四个关键帧，固定 50 Hz、每转换 `0.50 s`、每关键帧 hold `0.12 s`。COM 从 Phase41 原生投影 minimum-jerk 移到左脚，随后保持左加载；右脚净空按同一固定曲线从 `0.25 mm` 升至 `12 mm` 再降回。每帧仅允许 root+lower15 局部修正。

## 结果

- 中间求解帧：`72/72` 硬可行。
- 输出帧：`100`。
- 最坏等式残差：`8.90e-10`。
- 最坏不等式余量：`+2.28e-12`。
- 最大局部修正 L2：`0.00455`。
- 3 帧目标函数在迭代上限停止，但硬约束均严格通过；`objective_converged` 与 hard feasibility 分开记录。
- lower15 joint-step p95/max：`0.01613 / 0.01807 rad`。
- root 水平加速度最大值：`3.241 m/s²`，低于固定 `4.0 m/s²` 门。
- 墙时 `3.17 s`，0 `mj_step`、0 GPU、0 训练。

## 意义与边界

这是当前路线第一次从官方 Stage250 稳定盆地出发，形成连续的

`双支撑中心 → 左侧加载 → 左单支撑/右脚摆起 → 右脚落地且仍左加载`

几何路径。它直接修复了 Phase23 的中间穿透和 Phase24 的 COM 瞬时跳变问题，支持“显式载荷转移必须先于接触切换”的机制判断。

它仍只是准静态 COM/contact 几何轨迹，不包含 COP、GRF、关节力矩、centroidal angular momentum 或真实 contact integration，因此不能作为 Gold teacher，也不能进 RL。下一步只允许在该冻结左半路径上做 centroidal force feasibility preflight；若力不可行，则停止在几何层，不跑 raw physics。

## 产物

- `phase27_left_half_continuous_contract.json`
- `run_phase27_left_half_continuous.py`
- `phase27_result.json`
- `tests/test_phase27_left_half_continuous.py`
