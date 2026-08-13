# Phase33–35 — X2 原生稳定盆地上的短时域 DSMS

## 最终裁决

`NATIVE BASIN REPRESENTATION PROVEN / SOFT-CARTESIAN DSMS SHORT SOLVE REJECTED`

这组实验直接回应“稳定迈步是否早已提供”：是。Phase34 closed AimDK 的稳定直行就是本轮唯一初始盆地；没有再使用 PHUMA lunge 失败初态或人工静止 pose。

## 表示门

1. **Phase33 raw torque 18-knot spline**：机器人仍安全，但相对原始记录的节点 state tangent 最大误差 `0.8005`，不合格。
2. **Phase33b equivalent official PD target 18-knot spline**：qpos 误差改善至 `0.00431 rad`，但节点速度误差仍为 `0.8082`，不合格。
3. **Phase34 dynamically consistent shooting nodes**：用压缩 PD target 的实际 rollout 作为 shooting nodes，Phase34 closed state 只作为 soft reference。这样 1350 个约束的 absmax=`7.77e-15`，初始 state exact，root-z min=`0.6541m`、tilt max=`0.2837rad`，NLP 构造、objective、gradient 全部 finite。

关键认识：50 Hz 控制参数化可以保持稳定盆地，但不能逐节点复刻 1 kHz qvel；多重打靶必须用控制压缩后的动力学 rollout 作为零-defect 初值，而不是强塞记录状态。

## 唯一 Phase35 求解

- horizon `0.34s`，17 nodes × 20 substeps；
- 18 个 linear PD-target knots；
- 1 CPU thread、nice10、MUMPS、max_iter=50；
- 0 GPU、0 retry、0 参数扫描；
- exact native state soft tracking + right-foot 12mm clearance/contact objective。

IPOPT 运行 `254.3s` 后达到 iteration limit：last iterate objective `10.09`，但 constraint violation `1.068`；keep-best merit 正确返回 iteration0 的零-defect原生初值，而不是不可行的低 cost 迭代。

| 硬门 | 结果 | 判定 |
|---|---:|---|
| solver status 0/1 | `-1` | fail |
| max defect | `2.33e-14` | pass |
| root-z min | `0.6541m` | pass |
| root tilt max | `0.2837rad` | pass |
| right swing-off | `0ms` | fail |
| right clearance | `0mm` | fail |
| left contact fraction | `0.926` | fail |
| left foot speed p95 | `0.354m/s` | fail |
| terminal right contact 40ms | true | pass |
| joint semantic RMS | `0.0364rad` | pass |

## 结论

Phase34 原生稳定日志已经解决了“成功动力学盆地从哪里来”的问题；它不是当前缺口。当前缺口是一个能在保持 defect/contact feasibility 的同时改变接触模式的生成器。

本轮否定的是固定配置：`soft ankle Cartesian target + 18-knot PD spline + generic MUMPS/IPOPT`。它的梯度把迭代推出动力学流形，而不是沿可行流形形成 liftoff。按照预注册，不增加迭代、不扫 tracking weight、不把 iteration0 冒充 teacher。

下一路线应改结构，而不是改预算：

- 在 shooting variables 中显式加入 contact/COP/centroidal force 或 continuation/homotopy；或
- 学习式 privileged physical generator，但必须从 Phase34 native state/action 分布启动，而不是从不可行 GMR 冷启动。

训练、长时域 DSMS 和 teacher 导出均保持锁定。

## 产物

- Phase33/33b/34/35 contracts、runners、result JSON；
- `phase35_native_dsms_candidate.npz`（拒绝证据，不是 teacher）；
- 对应四个测试文件。
