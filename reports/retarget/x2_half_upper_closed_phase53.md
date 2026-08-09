# WBT Phase53：half-amplitude / half-slew closed 诊断

## 假设

若 Phase50 的问题主要是上肢目标幅值/速度超过冻结 Stage250 后端的吸扰范围，那么把 upper scale、excursion、slew 同时减半，应在保留实质上肢跟踪的同时显著回收横漂和航向偏差。

## 干预 / 对照

- A：immutable BASE Phase34。
- B1：immutable Phase50，scale=0.25、±0.12rad、0.20rad/s。
- Bhalf：唯一新 episode；scale=0.125、±0.06rad、0.10rad/s。其他 hash/PD/command/timing 全冻结。
- 新 physics episode=`1`，重试=`0`；未训练、未扫其他幅值。

## 结果

| 指标 | A/counterfactual A | B1 | Bhalf | Phase53门 | 通过 |
|---|---:|---:|---:|---:|:---:|
| full gate | True | True | True | True | True |
| half-target upper RMSE | 0.04166 | - | 0.03767 | 改善≥10% | False |
| half-target upper p95 | 0.09955 | - | 0.09047 | 改善≥10% | False |
| signed lateral (m) | -0.08361 | -0.15062 | -0.13628 | excess回收≥25% | False |
| signed yaw progress (rad) | 0.07575 | -0.19428 | -0.14786 | excess回收≥25% | False |
| stop drift (m) | 0.06726 | 0.09071 | 0.07619 | ≤B1+0.02 | True |

- upper RMSE/p95实际改善：9.58% / 9.12% 。
- lateral/yaw excess回收：21.40% / 17.19% 。
- Bhalf left/right slip p95：0.356/0.617m/s；contact保护门=`True`，slip保护门=`True`。
- Bhalf 1kHz telemetry alignment strict max gate：`False`；因此contact只作诊断，不改变上述telemetry核心否证。

## 结论

**STOP_UPPER_AMPLITUDE_ROUTE**

half幅值保持全功能门并改善部分contact/slip，但upper RMSE/p95只改善9.58%/9.12%，横漂/yaw只回收21.40%/17.19%，均未达到预注册10%/25%门；幅值路线停止，不扫0.25/0.75。

## 下一步

停止upper幅值扫描；后续若继续WBT，应转向具有独立重复基线的闭环扰动建模或允许下层受限协调，而不是继续缩放上肢。

接触/滑移为 official MuJoCo 模型真值，不是实机 GRF/COP。
