# X2 动态重定向赛马 A：Contact-aware DSMS

日期：2026-08-11
裁决：**MECHANISM_IMPROVED_BUT_REJECTED**
训练/PPO：0；solver/config scan：0。

## 单变量

继承 Phase3b 已通过的官方 X2 动力学、逐关节 PD、Euler、50 Hz reference velocity 与 raw-motor replay 合同。唯一变化是在冻结的 0.34 s 双支撑前缀内，将左右踝 EE 目标改成：

- sole XY 锁定各自初始 anchor；
- sole clearance 目标 0.25 mm；
- ankle linear velocity 目标为 0；
- 固定权重 xy/z/v = 100/300/20。

最大 reference target translation 为 24.96 mm；未扫描权重、clearance、solver、窗口或门槛。

## NLP

- IPOPT/MUMPS 300 iterations，515.71 s。
- 最终 status `-1`（maximum iterations exceeded）；dual infeasibility 15.26。
- keep-best 为 iteration 277：max defect `0.0011539`、defect p95 `3.19e-05`。
- 对比 Phase3b max defect `0.0542203`，动力学连续性改善约 97.9%，并通过 `<=0.01` 子门。
- 但 solver status 硬门未过。

## raw official motor replay（0.34 s）

| 指标 | Phase3b | Phase3c | 门/裁决 |
|---|---:|---:|---|
| max defect | .05422 | .001154 | Phase3c PASS |
| full-prefix survival | pass | pass | PASS |
| flight fraction | .2529 | .1529 | 改善，FAIL <=.02 |
| stance speed p95 L | .1300 | .0686 m/s | 改善，PASS |
| stance speed p95 R | .1906 | .1561 m/s | 改善，FAIL |
| excursion L/R | .0249/.0184 | .00774/.01223 m | PASS |
| active-sole penetration L/R | -3.54/-5.03 | -1.81/-4.76 mm | 改善，仍 FAIL |
| root accel p95 | 11.71 | 10.32 m/s² | 改善，FAIL |
| head max | .00036 | .02585 rad | FAIL |
| torque saturation | 1.689% | 1.338% | 改善（诊断） |

## 结论

显式 stance EE 目标是真正有效的方向：它大幅闭合 shooting defects，并改善 active-sole penetration、足速、flight、excursion、root acceleration 和 saturation。但软 ankle-body target 不能保证 official sole nonpenetration/contact，也不能同时约束右脚与 head 动态响应，因此不是合格 teacher。首次报告的 `-29.4/-49.5 mm` 混入每脚一个 `contype=0` visual mesh，已按官方 12 个 active sole spheres 更正；拒绝裁决不变。

按预注册停止本路线，不增加权重或迭代数。赛马转入 raw rollout 的 SBTO/CEM 与 DDR-style 搜索。

## 文件

- `phase3c/prereg_phase3c_contact_aware.json`
- `phase3c/phase3c_preflight.json`
- `phase3c/phase3c_solve_result.json`
- `phase3c/x2_lunge_phase3c_contact_prefix.npz`
- `phase3c/phase3c_raw_motor_replay.json`
