# X2 Native Subscriber-Receipt Event Replay Phase15

- 裁决：**PHASE15_RECORDED_CONTROL_PRESCRIBED_REJECTED**。
- reference/评分始终是 Phase14 50Hz recorded actual q/root/model-contact；recorded command 只作 control input。
- contact 是模型重建/官方MuJoCo碰撞，不是实机 GRF、COP、wrench；source trace稳定不冒充replay稳定。

## 假设

若 Phase12 失败主要来自 actual-q target + synthetic fixed-PD 控制合同不匹配，那么按 Phase14 subscriber receipt 时序回放 source q/Kp/Kd 应显著改善 prescribed-root trackability。

## 干预 / 对照

- 干预：逐1ms处理所有到期event，严格使用保存的global receipt order；changed+valid mask只更新active29 q/Kp/Kd，head保持source inactive。
- 历史对照 Phase12（不同8s源clip，仅作方法对照，不能当matched trajectory因果A/B）：q RMSE `0.1768`，body rel-pos p95 `0.2576m`，contact `0.457`，slip p95 `1.247m/s`。
- 初始化：首snapshot q/dq/root pose/root velocity；首snapshot前最后有效command context。缺失publisher timestamp、solver/contact warmstart。
- 先 prescribed-root；仅过完全相同 Phase12 gate 才允许唯一一次 free-root。无时移/增益/滤波扫描。

## 结果

| mode | survival | q RMSE(all31/active29/head2) | body rel-pos p95 | contact | SS(ref/real) | slip p95 | torque sat |
|---|---:|---:|---:|---:|---:|---:|---:|
| prescribed_root_trackability | 19.960/19.960s | 0.1062/0.1089/0.0522 | 0.1716m | 0.815 | 0.270/0.206 | 0.622m/s | 0.0130 |

- event context：snapshot0前 `60` events；之后消费 `39986`；active joints `29`，head active `False`。
- prescribed gate：`False`，失败项 `['body_position_trackable', 'body_orientation_trackable', 'slip_bounded']`。
- free-root executed：`False`；prescribed未过门，按预注册规则停止。

## 误差归因边界

- recorded q/Kp/Kd 与subscriber receipt时序已用；若prescribed仍失败，不能再归因于synthetic fixed-PD本身。
- 仍无法排除：publisher真实时间与callback receipt抖动、首snapshot之前的solver/contact warmstart、原仿真进程内部状态，以及1ms离散replay对异步event的量化。
- 本结果不评价Any2Any、训练policy或真机部署。

## 结论

- 结果：recorded event q/Kp/Kd + subscriber receipt timing仍未通过prescribed trackability门；按规则未运行free-root。
- 结论：Phase12失败不能只归因于synthetic fixed-PD；剩余差异位于receipt-vs-publisher timing、初始化/warmstart或未保存仿真内部状态。
- 下一步：停止Phase15，不扫时移/增益/滤波；先由主线裁决是否值得采集publisher-stamped或sim-state checkpoint。
