# WBT Phase52：mirrored-upper closed 因果诊断

## 假设

若 Phase50 的横漂/航向偏差主要由 `AMASS-UPPER-001` 的左右方向性直接造成，那么只镜像 upper14 后，`Bmirror−A` 的 signed lateral 与 yaw 应相对 `B1−A` 翻转；其他控制合同保持不变。

## 干预 / 对照

- A：BASE Phase34 immutable Stage250。
- B1：Phase50 原 upper14，immutable。
- Bmirror：唯一新 episode；仅按冻结 X2 joint mirror contract 对 upper14 做左右交换，roll/yaw 取反，pitch 不变。
- mirror²误差=0；lower/waist/root/head direct path、Stage219、PD、command、时序均未改变。

## 结果

| 指标 | A | B1 | Bmirror |
|---|---:|---:|---:|
| full gate | True | True | True |
| upper tracking RMSE (rad) | 0.03744 | 0.03925 | 0.03940 |
| signed lateral (m) | -0.08361 | -0.15062 | -0.16117 |
| signed yaw progress (rad) | 0.07575 | -0.19428 | -0.17956 |

- lateral delta：B1−A=-0.06701m，Bmirror−A=-0.07756m；sign flip=`False`。
- yaw delta：B1−A=-0.27003rad，Bmirror−A=-0.25531rad；sign flip=`False`。
- left slip p95 A/B1/Bmirror：0.425/0.614/0.441m/s。
- right slip p95 A/B1/Bmirror：0.392/0.592/0.388m/s。

## 结论

**DIRECTIONAL_UPPER_CAUSALITY_NOT_SUPPORTED**

Bmirror与B1相对A的横漂和航向偏差均保持同一负方向且幅值接近，因此Phase50漂移不能归因于该upper轨迹的左右方向性。镜像把slip显著拉回A附近，却没有翻转漂移，说明足端滑移对upper侧别敏感，而系统性横漂/航向更像非方向性的上肢扰动、闭环初态/调度差异或下层吸扰能力边界。

## 下一步

冻结本次归因结果；不要据同一episode拟合补偿。若继续辨识，需要预先固定的对称/反对称upper panel与重复A噪声基线。

接触/滑移来自 closed official MuJoCo 1kHz 碰撞模型，不是实机 GRF/COP。
