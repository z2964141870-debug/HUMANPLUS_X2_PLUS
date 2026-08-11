# X2 Phase13：PHUMA lunge 支撑裕量全段审计

日期：2026-08-11  
状态：**100% SINGLE-SUPPORT COM OUTSIDE / CONTACT SCHEDULE REPAIR REQUIRED / 0 PHYSICS**

## 问题

Phase12 发现首个右单支撑转换要求在固定双足下横移 COM 约0.40m，局部关节解会严重越限。本阶段把审计扩展到完整 Phase30 lunge，判断这是首段偶然状态，还是整个 X2 contact intent 的系统性矛盾。

## 合同

- 输入：冻结的 `PHUMA-LUNGE-R-001` Phase30 1.46x，175帧/30Hz。
- contact intent：冻结的 Phase29 source-height stance phases，不重新从当前几何推标签。
- 支撑面：意图支撑脚的 official active12 sphere center 凸包，向外扩5mm sphere半径。
- 质心：official X2 模型 subtree COM 的水平投影。
- 门：单支撑 COM-outside 帧占比 `<=5%`。
- 只作准静态几何诊断；它不是 realized collision、GRF、COP，也不能单独证明某个动态帧必倒。
- 175次 `mj_forward`，0 `mj_step`、0 optimizer、0 GPU。

## 结果

| 指标 | 数值 |
|---|---:|
| 总帧数 | 175 |
| 单支撑 / 双支撑 / flight intent | 137 / 38 / 0 |
| 单支撑 COM-outside | **137/137（100%）** |
| outside gap p50 / p95 / max | **0.214 / 0.392 / 0.431 m** |
| 左支撑 outside | 114/114（100%） |
| 右支撑 outside | 23/23（100%） |
| 两足中心距离 p50 / p95 / max | **0.801 / 0.978 / 0.994 m** |

右支撑段的 outside gap 中位为 `0.325 m`，左支撑段为 `0.180 m`；不是单侧命名或镜像造成的孤立错误。

## 解释

人体/G1 lunge 的宽足位被按名映射到 X2 后，stance schedule 仍要求长时间单脚承重，但 X2 COM 没有随支撑脚移动。动态动作允许 COM 短时离开静态支撑面，因此不能把单帧 outside 当作不可行证书；然而这里具备三项共同证据：

1. 137/137 单支撑帧全部 outside；
2. 缺口是20–40cm量级，而不是毫米级边界误差；
3. 既有 raw physics 同时表现为无稳定单支撑、root acceleration高和快速倾倒。

因此当前最可信根因不是训练/S2R，也不只是 PD teacher 不够强，而是**X2足位、COM路径与继承的contact schedule没有联合重定向**。

## 裁决

`ORIGINAL STANCE LABELS NOT ADMISSIBLE AS HARD X2 CONTACT TRUTH`

- 不再把 Phase29/30 的 source-height stance labels 当成动力学优化的硬接触真值。
- 不运行 Phase12 固定足位 task-space controller。
- 不再扫 Phase8/11 的 joint-mode CEM。
- teacher/强化学习继续锁定。

下一方法必须联合生成：X2 foot placement、contact schedule、root/COM path。它可以保留人体关键点/上肢语义，但应允许下肢和root显著偏离 Bronze；这更接近 DDR/contact-first 或 OmniTrack physical generator 的正确职责，而不是对现有 Bronze 做小 residual 修补。

## 产物

- `audit_phase13_support_margin.py`
- `prereg_phase13_support_margin.json`
- `phase13_result.json`
- 本报告
