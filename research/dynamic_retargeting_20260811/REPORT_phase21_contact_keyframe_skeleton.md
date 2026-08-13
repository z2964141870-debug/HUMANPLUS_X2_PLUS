# X2 Phase21：原生边界 contact-keyframe skeleton

日期：2026-08-11
状态：**SLSQP REJECTED / NO PHYSICS / NO TRAINING**

固定7 knots `[0,26,61,87,122,148,174]`、`DS-L-DS-R-DS`、Phase41同一原生安全首尾边界、官方active12/limits。PHUMA仅提供上身/时长软语义。

两项实现更正均发生在有效求解前并保留provenance：首次把Phase15错误当±0.65rad硬中心，违反“initialization only”；随后首次SLSQP在首评估因84行等式中8行线性冗余触发singular-C。修正为lower仅official limits，并用x0 rank-revealing QR固定76行独立等式；完整84行仍在最终复核。没有改变模板、权重、阈值或150轮预算。

唯一有效求解跑满150轮、831次函数评估，status9 iteration limit。独立/完整等式最大残差均0.8103，ineq min -0.305mm，未得到关键帧可行证书。dense 175帧复核更差：contact p95 0.697m、COM margin -0.557m、stance speed 1.010m/s、excursion 0.742m、flight 98.9%、upper p95 0.353m；仅qstep/rootacc/head等少数门通过。

因此7-knot单阶段SLSQP不是可用teacher，也不值得通过更多迭代或换初值继续。Phase19的逐帧局部可行并不能自动组成同时满足原生首尾和整段anchor连续性的全局轨迹。

下一步若继续，必须先单独求解离散 foot-placement/contact skeleton（不带PHUMA upper和原生终点），获得可行足端/COM骨架后再分阶段加入边界与语义；当前单阶段路线停止。

裁决：`SINGLE_STAGE_NATIVE_BOUNDARY_KEYFRAME_NLP_REJECTED`。
