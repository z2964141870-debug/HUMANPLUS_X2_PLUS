# BASE Phase42 — Stage250 原生低速 coherent boundary

## 裁决

`QUIESCENT 1S-SAFE NATIVE BOUNDARY SELECTED / NO PHYSICS OR TRAINING`

Phase41 按 PHUMA 姿态接近度选出的原生边界未来 1 秒安全，但 qvel 最大值 `5.074 rad/s`，与后续静止/准静态参考不连续。Phase42 不改 Phase34 轨迹，只在同一 650 个 coherent 1s-safe anchors 中按预注册确定性规则选择：full generalized qvel L2 最小，其次 max-abs，最后最早 trace index。

## 结果

- 选中 trace index `630`，stage=`stop`，elapsed=`6.40 s`。
- full qvel L2：`0.02084`，Phase41 为 `7.30860`。
- qvel max-abs：`0.01520`，Phase41 为 `5.07418`。
- root z：`0.65031 m`；tilt：`0.12272 rad`。
- 双支撑 COM margin：`+73.33 mm`。
- joint-limit overshoot：`0`。
- active12 min signed distance L/R：`-0.271 / -0.324 mm`，属于 Phase41/Stage250 原生 soft-contact 边界，不冒充严格 `-0.01 mm` contact seed。
- 0 `mj_step`、0 optimizer、0 GPU。

## 意义

这不是新的走路数据，而是从已有 Stage250 成功闭环中挑出的更一致初态。它适合下一次 DSMS/准静态路径的起点，因为“原生稳定状态 + 近零速度”同时成立；Phase41 仍保留为姿态接近 PHUMA 的历史对照。

下一步应以 Phase42 为唯一新变量，重新跑冻结的 centered load-transfer 几何管线；通过后才能构造 DSMS 短前缀。不得把 Phase27（围绕 Phase41 构建）直接拼到 Phase42。

## 产物

- `tools/official_x2/audit_phase42_native_quiescent_boundary.py`
- `reports/official_x2/phase42_native_quiescent_boundary.json`
- `tests/test_phase42_native_quiescent_boundary.py`
