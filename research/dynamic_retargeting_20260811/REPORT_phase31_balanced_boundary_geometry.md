# Phase31 — X2 平衡低速边界载荷/摆脚几何

## 裁决

`BALANCED BOUNDARY STILL REJECTED / STOP NATIVE-BOUNDARY SELECTION ROUTE`

唯一变量为 Phase43 预先选择的低速且姿态较近原生边界；Phase26 几何合同全部冻结。

- `DS_CENTER`：原生接受。
- `DS_LOAD_LEFT`：硬可行，等式 `1.58e-10`。
- `L_SUPPORT_R_SWING`：严格失败；等式 `1.34e-6 > 1e-6`，不等式通过。
- 失败构型左 ankle roll 再次到官方上限 `0.2625 rad`。
- 0 `mj_step`、0 GPU，墙时 `1.01 s`。

Phase42 最小速度边界与 Phase43 低速/姿态平衡边界都在同一“COM 精确移到足中心后抬对侧脚”步骤碰 ankle limit；继续挑 Stage250 锚点会转为结果驱动搜索。当前更合理的结论是准静态 `COM xy = support-foot centroid` 等式过强：真实动态单支撑允许 COM 不在足中心，通过 COP、接触力与有限 COM acceleration 满足平衡。

因此停止 native-boundary selection 路线。下一生成器应直接联合 COP/GRF/centroidal dynamics，或使用 DSMS contact-implicit shooting；不得继续扫描边界或放宽踝关节限位。

## 产物

- `phase31_balanced_boundary_geometry_contract.json`
- `run_phase31_balanced_boundary_geometry.py`
- `phase31_result.json`
- `tests/test_phase31_balanced_boundary_geometry.py`
