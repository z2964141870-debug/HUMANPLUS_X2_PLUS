# X2 Native Gold Official Trackability Phase12

- 裁决：**PHASE12_PRESCRIBED_TRACKABILITY_REJECTED**。
- 唯一 reference 是 Phase11 MotionLib+state adapter 输出的 recorded actual q；recorded command 未作 reference。
- source trace 自身稳定不等于本次 replay 稳定；prescribed-root 也不证明平衡。
- contact 是模型几何/官方仿真碰撞，不是实机 GRF、COP、wrench 或足底力真值。

## 假设

目标本体原生 actual-q Gold 若能通过同一官方模型的固定-root trackability，至少可作为训练管线 sanity reference；free-root 裸PD只额外检验无闭环策略时的开环平衡。

## 干预 / 对照

- 对照：Phase10 recorded source trace（仅作 reference/provenance，不冒充 replay）。
- 干预A：官方 AimDK v1.0 scene、1kHz physics/50Hz official PD，root pose/velocity逐物理步外部 prescribed。
- 干预B：A过门后，同一 reference、同一初始化只跑一次 free-root；无参数扫描。
- 初始化：q、dq、root pose、root lin/ang velocity均取 frame0 actual；solver/contact warmstart和原ONNX hidden state不可得。

## 结果

| mode | survival | q RMSE | body rel-pos p95 | root XY RMSE | contact agreement | SS(ref/real) | slip p95 | torque sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prescribed_root_trackability | 7.980/7.980s | 0.1768 | 0.2576m | 0.0000m | 0.457 | 0.120/0.484 | 1.247m/s | 0.0007 |

- prescribed gate：`False`，失败项 `['body_position_trackable', 'body_orientation_trackable', 'contact_agreement', 'slip_bounded']`。
- free-root executed：`False`；因 prescribed 未过门而停止。

## 结论

- 结果：prescribed-root 未通过预注册 trackability 门；按规则停止，未运行 free-root。
- 结论：Gold/PD replay contract 仍有不兼容；该结果只否定此裸PD sanity replay，不评价训练策略。
- 下一步：停止本阶段；先核查失败指标，不扫参数、不训练。
