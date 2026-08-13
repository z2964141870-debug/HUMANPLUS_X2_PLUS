# X2 Phase19：分层硬接触生成器可行性预检

日期：2026-08-11
状态：**LOCAL HARD GEOMETRY 175/175 PASS / TEMPORAL GENERATOR UNLOCKED / PHYSICS LOCKED**

## 问题

Phase18 的加权 GN 能把 COM 缺口缩到毫米级，却无法同时满足接触、足速、平滑和上身语义。本阶段不再调权重，只判断把 contact/COM/swing 改成硬约束后，单帧 X2 几何是否本身无解。

## 合同

- 固定 Phase18 最接近的 `DS-L-DS-R-DS` 模板，不改时序。
- 每帧变量：root XYZ、root local rotation、lower12+waist3，共21维。
- 硬等式：所有 active sole signed distance=0；COM XY=当前支撑足中心。
- 硬不等式：swing sole signed distance≥12mm。
- 使用 Phase18 冻结 correction bounds，并叠加官方 joint limits。
- 每帧仅求一个 HiGHS 线性可行证书；不最小化动作误差，不执行 `mj_step`。

## 结果

- 总帧：175；可行：**175/175**。
- 单支撑：69；可行：**69/69**。
- 双支撑：106；可行：**106/106**。
- 最大使用 correction：0.65（触及冻结 lower-q correction bound）。
- 175 次 LP、175 次 FK，0 physics、0 policy、0 optimizer、0 GPU。

## 解释

这排除了“X2 在给定模板下连单帧接触几何都做不到”的判断。Phase18 失败来自软代价之间的折衷和跨帧耦合，而不是所有帧的局部可行域为空。

但这还不是 teacher：逐帧可行解可能在相邻帧间跳变，也没有约束 stance speed、excursion、root acceleration、力/矩或物理稳定性。下一步只解锁 **temporal hard-constraint hierarchy**：先保持这些硬接触约束，再最小化跨帧速度/加速度和语义误差；physics 与训练继续锁定。

当前裁决：`LOCAL_HARD_CONTACT_FEASIBLE / TEMPORAL_CERTIFICATE_REQUIRED`。
