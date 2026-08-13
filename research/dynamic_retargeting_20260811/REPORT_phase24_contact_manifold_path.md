# Phase24 — X2 连续接触流形路径

## 裁决

`FAIL_AT_FIRST_LOAD-TRANSFER FRAME / EXPLICIT COP-LOAD TRANSFER REQUIRED`

按预注册单一配置，从 Phase22 七关键帧与 Phase23 固定 50 Hz 时间轴出发，对每个中间帧求解 root+lower15 局部修正。支撑脚接触和 XY 锚点、COM 支撑约束、全足非穿透、摆脚净空渐变、关节限位均为硬约束；没有 physics、GPU、训练或第二配置。

## 执行结果

- 第 0 段 `DS_NATIVE → DS_LOAD` 的 24 个中间帧可行。
- 第 1 段 `DS_LOAD → L_SUPPORT_R_SWING` 的第 1 帧失败，时间 `u=0.04`（20 ms）。
- 当帧右摆脚净空目标仅 `0.257 mm`，不等式余量为 `+1.610 mm`，因此失败不是抬脚高度过大或穿透。
- 硬等式最大残差 `9.046 mm`；SLSQP 达 80 次上限，修正 L2 `0.782`，仍不能闭合。
- 总计尝试 25 个中间帧，墙时 `1.16 s`，`mj_step=0`。

## 根因解释

Phase22 的 `DS_LOAD` 名称高估了其物理含义：该关键帧仍将 COM 水平投影约束在双脚质心中心，并未表示载荷向左脚转移。下一帧一旦把活跃支撑集改成仅左脚，当前模型就把 COM 目标瞬间切到左脚中心，造成不连续的约 `0.137 m` 横向目标变化；在单个 20 ms 帧和冻结边界内不可行。

因此：

- Phase22 证明的是离散几何姿态存在，不是可连续切换的支撑力序列。
- Phase23 的中间穿透和 Phase24 的首帧失败都不能靠延长时间单独解决。
- 继续做纯 IK/contact projection 会重复遇到同一载荷切换缺口。

下一步必须先增加显式的双支撑 load-transfer 层：COM 目标只需留在支撑多边形内，COP/左右法向载荷从双支撑连续转移到单支撑，并给定有限过渡时间；该层通过后再恢复逐帧接触流形与摆脚。此结论不授权 physics 或 RL。

## 产物

- `phase24_contact_manifold_path_contract.json`
- `run_phase24_contact_manifold_path.py`
- `phase24_result.json`
- `tests/test_phase24_contact_manifold_path.py`
