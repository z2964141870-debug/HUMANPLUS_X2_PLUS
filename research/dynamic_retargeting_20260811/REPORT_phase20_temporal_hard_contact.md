# X2 Phase20：跨帧硬接触零空间生成器

日期：2026-08-11
状态：**VALID RUN REJECTED / NO PHYSICS / NO TRAINING**

## 合同与实现

固定 `DS-L-DS-R-DS` 模板。每帧先计算 contact/COM/swing 的最小硬投影，再在17–18维局部零空间中用全局 sparse LSQR 降低 stance-foot velocity、state velocity/acceleration 和语义修正。10轮、trust region 0.10、单线程、0 GPU，不扫描参数。

第一次内部运行把 hard projection 与 null-space step 一起缩放，违反“硬约束优先”，因此标记 `INVALID_PRIORITY_SCALING`，不进入裁决。修正只改变层级执行顺序：hard projection优先占用trust budget，剩余部分才允许null-space smoothing；权重、模板、阈值和迭代数不变。

## 唯一有效结果

- 10轮，wall time 5.67s，`mj_step=0`。
- hard violation score：509.6→394.9→288.6→184.5→92.9→37.5→28.1→24.3→23.7→27.9。
- 最终 COM margin `+13.4mm`，已经转入几何支撑面。
- 最终 stance contact p95 `13.96mm`（门0.5mm）。
- stance speed p95 `0.316m/s`（门0.10）。
- stance excursion `0.164m`（门0.03）。
- swing clearance `5.69mm`（门12mm）。
- root horizontal acceleration p95 `12.90m/s²`（门4）。
- upper keypoint p95 `0.236m`（门0.10）。
- limits、qstep、head lock通过；完整门失败。

## 结论

Phase19证明每帧局部可行，但Phase20证明在冻结10轮/0.10 trust合同内，从Phase15初值投影到连续硬接触流形仍不够：COM先修复，而contact、足速、root acceleration与upper语义继续冲突。不能通过追加迭代或增大trust把当前近似当teacher。

这不否定分层硬约束方法；它否定的是“从Phase15整段初值用10轮局部线性投影直接到达可行轨迹”。下一表示需要先构造低维接触关键帧/foot-placement knot 的全局可行骨架，再做密集时间插值，而不是175帧同时从坏初值推进。

裁决：`DENSE_FROM_BAD_INIT_TEMPORAL_PROJECTION_REJECTED / KEYFRAME_CONTACT_SKELETON_REQUIRED`。
