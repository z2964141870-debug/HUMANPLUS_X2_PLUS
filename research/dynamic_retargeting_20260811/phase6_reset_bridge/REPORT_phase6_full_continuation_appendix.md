# Phase 6 附录：Projected reset + frozen bridge 的 5.8 s continuation

日期：2026-08-11
性质：**纯重放；0 优化、0 搜索、0 调参；不覆盖 Phase6 原判定**。

## 目的与冻结合同

本附录只回答一个问题：Phase6 projected reset 是否仅能通过首个 `0.34 s`，还是能实质改善原始 free-root 的 `0.575 s` 生存时间。

- 直接读取既有 `phase6_projected_initial_state.npz`，没有重新投影。
- qvel 继续使用已保存的全零 reset。
- `0–0.34 s` 使用原来的 smoothstep bridge；此后完全回到原 Phase30 reference。
- official raw torque motor、official per-joint PD、MuJoCo 3.3.7、Euler/1 kHz 均不变。
- 完整运行 5.8 s，但接触、滑移和 root 的因果诊断只统计首次跌倒前。
- 三次相同重放的首次跌倒均为 `0.683 s`，结果确定性复现。

## 结果

| 指标 | 结果 |
|---|---:|
| 原始 free-root 首次跌倒 | 0.575 s |
| projected reset + frozen bridge 首次跌倒 | **0.683 s** |
| 延长 | +0.108 s |
| 生存比例 | 1.188×（+18.8%） |
| 跌倒触发 | tilt `0.90059 rad > 0.90` |
| 跌倒时 root-z | 0.54673 m |
| 跌倒前 root XY 净位移/最大 excursion | 0.17309 / 0.17309 m |
| 跌倒前 root horizontal accel p95/max | 4.4105 / 4.4365 m/s² |
| 跌倒前 torque saturation | 0.0472% |

因此它并非只过首屏：生存时间确有可重复的 `108 ms` 改善。但改善量不足以改变“完整动作失败”的结论，跌倒机制从初态接触异常进一步收缩为后续倾倒/支撑转换失败。

## Active 12-sphere contact/slip（跌倒前）

| 指标 | Left | Right |
|---|---:|---:|
| contact fraction | 100% | 100% |
| 意图支撑时实现接触 | 100% | 100% |
| 意图支撑时矛盾 | 0% | 0% |
| 非意图期仍接触 | 100% | N/A（跌倒前一直是意图支撑） |
| min clearance | -1.063 mm | -1.038 mm |
| slip p95 | 0.0344 m/s | 0.0329 m/s |
| slip max | 0.0526 m/s | 0.1127 m/s |
| contact switches | 0 | 0 |

双脚从 reset 到跌倒始终接触且没有 flight。尤其左脚在 source 非意图接触区间仍保持 100% 接触，没有发生期望的卸载/转换。结合 `0.173 m` 的 root 水平漂移和倾角触发，结果更符合“低滑移双脚约束下，身体继续向前/侧方倾倒”，而不是脚底失联造成的自由落体。

## 裁决

1. **Reset projection 是必要的公共前置层，但不是充分解。** 它干净地修复了 t0 双脚接触并延长 `18.8%` 生存时间。
2. **后续核心应是接触切换与质心/角动量控制。** 现有 bridge 只是逐渐撤销关节偏移，无法主动卸载左脚或约束 root 倾倒。
3. **不追加本路线参数。** 本附录没有改变 Phase6 的 `7/8 gates, overall FAIL`；动态软穿透仍约 1 mm，完整动作仍在 0.683 s 倒下。

## 新增产物

- `x2_lunge_phase6_full_continuation.py`
- `phase6_full_continuation.json`
- `REPORT_phase6_full_continuation_appendix.md`
