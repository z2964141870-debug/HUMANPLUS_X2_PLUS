# X2 WBT Canonical Root-Ground/Contact Contract Phase7

- 裁决：**PHASE7_CANONICAL_GEOMETRY_VALID_DYNAMIC_NOT_PROMOTABLE**。
- 回到 original source generator；没有 source-time、foot offset、teacher、policy、训练、checkpoint 或真机干预。
- MuJoCo collision/contact、COM、DCM 是官方模型估计量，**不是实机 GRF、COP 或足底力真值**。

## 官方机器人级常量与公式

- `x2.xml` reset root-z：`0.680000m`。
- 24 个官方足底碰撞球，半径：`0.005000m`；reset 最低 signed clearance：`0.005050000m`。
- 唯一 root-z 公式：`z'=z+MA9(c_reset-min_24_official_sole_signed_distance(q,root))`。MA9 继承 source generator 已固定的 smooth-window，不由五条动作结果选择。
- 模型 contact intent：单脚最低球距离 `<= c_reset + sphere_radius`；official realized contact 则来自 prescribed MuJoCo collision。二者都不是硬件力真值。

## Phase4–6 符号分裂根因

raw `gmr` 将不同 AMASS actor/sequence 的绝对 pelvis 高度带入 X2：KIT walk root-z 中位数约 0.695m，四条 ACCAD 约 0.614–0.626m；同一 X2 因而一条悬空、四条穿地。canonical 公式删除该绝对高度泄漏，但保留 joint、root XY/orientation 和 FPS。

另发现 Phase4–6 signed-distance helper 把 ankle-roll 的 non-contact visual mesh 与 12 个 active sole spheres 混在一起；这污染旧几何距离数字，但不影响 MuJoCo 实际 collision replay。Phase7 已只保留每脚 12 个 active sphere。

## 固定五动作 paired gate

| role | raw root-z p50 | correction min..max / step max | ground err p95 | contact agree/F1 | original→canonical survival | Δ(s) | slip/limit | paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walk_straight | 0.695 | -0.016..0.006/0.0035 | 0.0058 | 0.362/0.006 | 1.557→1.541 | -0.016 | 0.0095/0.0301 | True |
| turn_left | 0.618 | 0.021..0.072/0.0063 | 0.0150 | 0.633/0.205 | 1.890→1.942 | +0.052 | 0.1545/0.1690 | True |
| turn_right | 0.614 | 0.021..0.057/0.0056 | 0.0127 | 0.625/0.198 | 1.117→1.142 | +0.025 | 0.0979/0.1187 | True |
| stand_to_walk | 0.626 | 0.024..0.072/0.0064 | 0.0107 | 0.556/0.075 | 1.603→1.636 | +0.033 | 0.0080/0.0259 | True |
| walk_to_stand | 0.625 | 0.019..0.068/0.0062 | 0.0130 | 0.494/0.141 | 1.952→2.684 | +0.732 | 0.1603/0.0871 | False |

## 裁决边界

- 结果：五动作 root geometry 与 survival 通过；slip 在 ['walk_to_stand'] 越门，prescribed realized-contact agreement 在 ['walk_straight', 'turn_left', 'turn_right', 'stand_to_walk', 'walk_to_stand'] 未过 Phase4 0.90 门。
- 结论：统一公式解决 source root-z 符号分裂，可冻结为 Bronze 几何契约；contact realization 与当前 open-loop reference 均不能直接晋级 dynamic Silver。
- 下一步：冻结公式与失败证据，不调 clearance/window；下一阶段应修 stance-foot 水平约束/接触时序，而非再改 root-z。
- Bronze canonical 几何通过不等于 dynamic Silver，更不等于可训练或可部署。当前结果只否定把该 open-loop reference 直接晋级 Silver；不否定 X2、Any2Any 或 WBT。
