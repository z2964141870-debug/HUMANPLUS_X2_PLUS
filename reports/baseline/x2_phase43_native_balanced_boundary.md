# BASE Phase43 — Stage250 低速/姿态平衡原生边界

## 裁决

`BALANCED LOW-SPEED 1S-SAFE BOUNDARY SELECTED / GEOMETRY GATE NEXT`

在 Phase34 的 650 个未来 1 秒安全锚点中，先施加与准静态拼接一致的硬门 `full qvel max <= 0.1 rad/s`，再按 lower15 到 Phase15/PHUMA 两端均值的 L2 距离最小选择；后续 tie-break 为 qvel L2、最早时间。没有读取任何边界上的几何 solver 结果来挑样本。

- 低速 eligible：`239/650`。
- 选中 trace index `468`，stop `3.16 s`。
- qvel max/L2：`0.04185 / 0.08432`。
- lower15 目标距离：`0.93063 rad`。
- root z/tilt：`0.64775 m / 0.12821 rad`。
- 双支撑 COM margin：`+73.80 mm`。
- joint-limit overshoot：`0`。
- active12 min signed L/R：`-0.272 / -0.306 mm`，仍属于原生 soft-contact 边界。
- 0 `mj_step`、0 optimizer、0 GPU。

Phase41 姿态较近但速度过高；Phase42 速度最低但局部摆脚不可达。Phase43 是预先定义的分层选择，不是事后按 solver residual 搜索。下一步只允许以它替换边界，运行一次完全冻结的 Phase26 centered load-transfer skeleton。

## 产物

- `tools/official_x2/audit_phase43_native_balanced_boundary.py`
- `reports/official_x2/phase43_native_balanced_boundary.json`
- `tests/test_phase43_native_balanced_boundary.py`
