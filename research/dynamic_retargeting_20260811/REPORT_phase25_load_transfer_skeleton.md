# Phase25 — X2 准静态载荷转移骨架

## 裁决

`LEFT HALF-CYCLE HARD-FEASIBLE / DIRECT LEFT→RIGHT LOAD SHIFT REJECTED`

该阶段只使用 COM 投影作为准静态载荷转移代理；不把它称为 COP、GRF、centroidal dynamics 或动力学真值。单进程、1 CPU thread、0 `mj_step`、0 GPU、0 训练。

## Provenance 修正

首次预检在任何求解前发现 `DS_CENTER` 实现错误：代码把中心定义成双足几何中点，而 Phase41 原生稳定边界的 COM-x 与该点相差 `20.86 mm`。该无求解结果判 INVALID。随后只把 `CENTER` 修为 Phase41 原生 COM 投影；左右加载目标、顺序、边界、阈值和 solver 均未改，执行唯一有效求解。

## 有效结果

| 关键帧 | 硬可行 | 最坏等式残差 | 最坏不等式余量 |
|---|---:|---:|---:|
| DS_CENTER | 是 | 0 | +0.194 mm |
| DS_LOAD_LEFT | 是 | 1.45e-8 | +1.49e-9 |
| L_SUPPORT_R_SWING | 是 | 1.39e-10 | -6.80e-11 |
| DS_R_TOUCHDOWN_LEFT_LOADED | 是 | 8.49e-11 | +9.99e-12 |
| DS_LOAD_RIGHT | 否 | 7.34e-5 | +1.46e-6 |

左加载、右脚稳定摆起和右脚落地均闭合。失败只发生在从左加载 COM 目标直接切到右加载目标：两脚横向中心相距约 `0.2743 m`，而预注册单步 root-xy 修正上限为 `0.25 m`。SLSQP 到 100 次上限后仍有 `73.4 μm` 等式残差，按 `1e-6` 硬门诚实判失败。

## 解释

Phase24 的首帧失败支持“缺载荷转移层”；Phase25 进一步证明该方向并非整体不可行：半个周期的加载—摆动—落脚已经存在硬可行解。当前序列的问题是把完整的左→右载荷迁移压成单个关键帧，而不是右侧支撑构型必然不存在。

下一阶段只能作为新假设显式加入 `DS_UNLOAD_LEFT_TO_NATIVE_CENTER`，再从中心到 `DS_LOAD_RIGHT`；不得放宽 `0.25 m` 单步边界，也不得把 `73 μm` near-miss 当通过。即使完整准静态骨架通过，仍需时间连续、COP/GRF/centroidal force 与 raw physics 复核后才能成为 teacher。

## 产物

- `phase25_load_transfer_skeleton_contract.json`
- `run_phase25_load_transfer_skeleton.py`
- `phase25_result.json`
- `tests/test_phase25_load_transfer_skeleton.py`
