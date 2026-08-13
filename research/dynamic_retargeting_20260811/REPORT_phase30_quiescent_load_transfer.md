# Phase30 — X2 低速原生边界载荷转移

## 裁决

`QUIESCENT BOUNDARY LOADS LEFT / SWING GEOMETRY REJECTED`

唯一变量为 Phase26 的 Phase41 姿态近邻边界替换成 Phase42 最小 qvel 原生边界；接触、COM 序列、边界、solver 和容差全部冻结。

- `DS_CENTER`：原生接受。
- `DS_LOAD_LEFT`：硬可行，等式 `2.23e-10`。
- `L_SUPPORT_R_SWING`：严格失败；等式 `1.97e-6 > 1e-6`，不等式 `-8.80e-7 < -1e-8`。
- 失败解左 ankle roll 达到官方限位 `0.2625 rad`，说明该低速停止姿态在冻结局部边界内缺少右脚抬起余量。
- 0 `mj_step`、0 GPU、墙时 `1.01 s`。

Phase42 将 qvel max 从 `5.074` 降到 `0.0152 rad/s`，但“只最小化 qvel”牺牲了与目标载荷/摆脚姿态的局部可达性。不能把 Phase27（围绕 Phase41）直接拼接到 Phase42，也不能放宽 ankle limit 或 solver 容差。

下一边界选择应在 650 个 coherent anchors 中同时使用预先定义的低速资格和 lower15 姿态距离，再只运行一次冻结几何门；不得按本次 solver residual 逐锚点挑最好结果。

## 产物

- `phase30_quiescent_load_transfer_contract.json`
- `run_phase30_quiescent_load_transfer.py`
- `phase30_result.json`
- `tests/test_phase30_quiescent_load_transfer.py`
