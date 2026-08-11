# Phase5 路线 C：DDR 式 rolling CEM 最小复现

## 裁决

本路线完成了预注册的唯一一次搜索，结果 **FAIL（有效负结果）**。它不是实现崩溃：raw official MuJoCo 预检和确定性检查通过，候选在独立 raw replay 中通过 8 个共同门中的 7 个；但 active-contact sole penetration 仍为 `L=-1.776 mm / R=-4.367 mm`，未达到 `>=-0.5 mm`。

因此，当前小规模 DDR 表示能够保持动作语义、存活和低足速，但没有解决核心的几何/接触穿透。按预注册停止，不追加 seed、样本数、权重或 horizon 扫描，不进入 RL/PPO。

> **审计更正（active12）**：初版共同 penetration auditor 将每脚一个 `contype=0` visual mesh（geom `14/37`）也按 `geom_size[0]` 球半径计算，产生了 `-30～-51 mm` 的错误量级。本报告现只统计每脚 12 个 `contype!=0` 的真实 sole sphere。未重新运行 CEM；仅确定性重放已经保存的 Bronze/candidate targets。旧结果保存在 `result_pre_active12_audit.json`，已由带 provenance 的 `result.json` 取代。

## 冻结合同

- 动作：`PHUMA-LUNGE-R-001`，Phase30 `1.46x`，前 18 个 50 Hz 节点（`0.34 s`）。
- 物理：X2 official raw motor scene，`motion_control.yaml` 逐关节外部 PD，Euler `1 kHz`，MuJoCo `3.3.7`。
- 搜索：seed `5505`；population `24`；elite `6`；每步 `3` 轮；4 节点滚动窗口；执行首节点；共 17 次决策。
- 控制量：31 维 PD target sequence，围绕 Bronze 初始化，head target 固定零。
- 主任务 cost：pelvis、双踝、双腕世界关键点，pelvis 姿态，以及支撑接触、支撑足速度、flight、root acceleration。**没有 Bronze joint-angle tracking 项**。

## Preflight

- scene/control/motion SHA 已写入 `preflight.json`。
- raw motor：通过。
- official Euler：通过。
- MuJoCo `3.3.7`：通过。
- Bronze/zero-offset replay 两次结果完全一致：通过。
- 正式搜索前 Bronze 本身未通过共同门，失败项为 penetration；这正是本路线要改善的基线。

## Raw replay 对照

| 指标 | Bronze | DDR-CEM | 门/方向 |
|---|---:|---:|---:|
| 存活 0.34 s | PASS | PASS | PASS |
| semantic keypoint RMSE | 37.83 mm | **34.87 mm** | `<=80 mm` |
| penetration L（active12） | **-1.776 mm** | -1.776 mm | `>=-0.5 mm` |
| penetration R（active12） | **-3.705 mm** | -4.367 mm | `>=-0.5 mm` |
| contact contradiction L | 0.00% | 0.00% | 诊断 |
| contact contradiction R | 12.35% | **10.59%** | 诊断 |
| stance speed p95 L | **0.0293 m/s** | 0.0457 m/s | `<=0.10` |
| stance speed p95 R | **0.0346 m/s** | 0.0387 m/s | `<=0.10` |
| stance excursion L（active12 centroid） | **0.00694 m** | 0.00928 m | `<=0.03` |
| stance excursion R（active12 centroid） | **0.01377 m** | 0.01513 m | `<=0.03` |
| unintended flight | 0.00% | 0.00% | `<=2%` |
| root accel p95 | **3.610 m/s²** | 3.743 m/s² | `<=4.0` |
| head max | **0.00191 rad** | 0.00440 rad | `<=0.02` |
| torque saturation | **0.00%** | 0.797% | 诊断 |

搜索把关键点 RMSE 改善约 `2.96 mm`、右脚接触矛盾改善约 `1.76` 个百分点，但代价是足速、足位移、root accel 和饱和率略变差。active12 穿透没有改善：左脚差异小于 `0.001 mm`，右脚反而恶化约 `0.662 mm`。

## 解释和边界

这次结果否定的是当前 **小样本、短 horizon 的 DDR rolling-CEM smoke**，不是 DDR 方法族。主要信息是：仅靠关键点、姿态及软接触代价，搜索倾向于维持 Bronze 附近的可追踪动作，却没有动力把真实约 `1.8–4.4 mm` 的 sole 嵌入修正到硬门范围；右脚甚至恶化。若未来继续 DDR，至少需要把 active sole signed distance/contact complementarity 从软诊断提升为候选淘汰或硬可行性层；直接扩大 population 并没有被本次证据支持。

## 产物

- `prereg_phase5_ddr.json`
- `phase5_ddr_x2.py`
- `preflight.json`
- `result.json`
- `result_pre_active12_audit.json`（被更正的旧 auditor 结果）
- `active12_audit.json`
- `audit_phase5_ddr_active12.py`
- `ddr_candidate.npz`
- 本报告

没有运行 RL/PPO，没有修改 Phase3b/Phase3c 或其他并行路线。
