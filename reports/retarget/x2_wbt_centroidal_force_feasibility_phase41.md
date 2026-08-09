# X2 WBT Phase41：centroidal / contact-force feasibility oracle

## 假设

Phase40 固定的25帧 DS→R-SS→DS 轨迹只有在 official X2 模型上存在满足六维平衡、接触激活和摩擦锥的 GRF，才值得进入动力学感知 alternating teacher。

## 干预

- 路径、30Hz、root orientation、contact intent、official `x2.xml` 全部冻结。
- MuJoCo 模型计算 COM、COM acceleration、centroidal angular-momentum rate。
- 每帧只允许 intended stance 且 official sole signed-distance 位于 [-0.01, 0.5]mm 的 sphere 承力；非接触脚 force=0。
- 接触力满足非负法向力、official μ=1.000 的保守L1摩擦锥、总力/总矩等式。非负法向力在实际sphere点的凸组合同时形成足底内COP。

## 对照与结果

- 有效导数帧：23；几何激活通过：0/23。
- force LP feasible：0/23；geometry+force共同可行：0/23。
- 不可行帧：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]。
- COM acceleration norm p95/max：[6.656709002783337, 8.469261436598904]m/s²。
- centroidal Hdot norm p95/max：[33.291944598852545, 37.23282516368459]N·m。

## 结论

**PHASE41_CENTROIDAL_FORCE_ORACLE_INFEASIBLE**

冻结Phase40局部轨迹未通过模型接触几何+六维GRF硬可行性层；当前不允许用soft reward或policy训练绕过，也不进入alternating teacher。

这些 GRF/COP/contact 都是 official MuJoCo 模型估计量，**不是 X2 实机足底力、COP 或动力学真值**。

## 下一步

按门停止；仅报告具体几何激活或力/矩平衡缺口，不调路径、contact schedule、摩擦或阈值。
