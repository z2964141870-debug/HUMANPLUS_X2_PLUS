# X2 Phase14：仅重标 contact schedule 的可行性审计

日期：2026-08-11  
状态：**LABEL-ONLY REPAIR FAILED / GEOMETRY MUST CHANGE / 0 PHYSICS**

## 假设

Phase13 的单支撑标签可能只是过于激进。如果保持 Phase30 的 q、root 和足位不变，仅把不满足准静态支撑的帧从单支撑改成双支撑，或许能得到一条低成本 contact schedule。

## 合同

- 每帧候选仅为左、右、双支撑三类。
- 原标签若 COM 支撑裕量非负则保留；否则选与原标签集合差最小的可行标签。
- 支撑面仍为 official active12 sphere center 凸包外扩5mm。
- q、root、foot placement 完全不改。
- 通过门：所有帧都有可行标签，且改标比例 `<=20%`。
- 0 physics、0 optimizer、0 GPU。

## 结果

| 指标 | 数值 |
|---|---:|
| 总帧 | 175 |
| 改标 | **137（78.3%）** |
| 仍无可行标签 | **30** |
| 修复后左单支撑 | 0 |
| 修复后右单支撑 | 0 |
| 修复后双支撑 | 145 |
| 原/修复 transition | 8 / 4 |
| 双支撑 margin min / p05 | **-24.8 / -21.1 mm** |

所有能靠标签修复的原单支撑帧都只能退化为双支撑；另外30帧即使把两脚都算支撑，COM投影仍在联合凸包之外。

## 裁决

`LABEL-ONLY CONTACT REPAIR REJECTED`

这不是阈值边缘问题：改标量达到78%，而且仍有厘米级双支撑缺口。把整条动作标成双支撑既会丢失 lunge 的支撑语义，也无法满足完整几何门。

因此下一方法必须联合修改：

- X2 足位/步宽；
- root/COM 路径；
- contact schedule；
- 必要时下肢 q。

上肢和人体关键点语义可以作为软跟踪项，但当前 Bronze lower/root/contact 不能继续作为小 residual 周围的硬中心。teacher、RL 与新的 physics rollout 继续锁定，直到联合几何生成器先过离线门。

## 证据边界

COM 支撑裕量是准静态诊断，不是 dynamic ZMP/GRF 证书；本阶段只证明“原几何上仅改标签”不成立，不证明某个联合轨迹优化器一定成功或失败。

## 产物

- `audit_phase14_contact_schedule_repair.py`
- `prereg_phase14_contact_schedule_repair.json`
- `phase14_result.json`
- 本报告
