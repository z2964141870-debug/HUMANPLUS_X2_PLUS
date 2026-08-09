# X2 Official Native Dynamic Gold Seed Phase10

- 裁决：**PHASE10_NATIVE_DYNAMIC_GOLD_SEED_EXPORTED**。
- 输入是 AimDK X2 v1.0 官方 MuJoCo + 随包 `kuailechongbai.onnx` 的 actual-state 舞蹈 trace；只读源数据、未训练、未重放、未修改物理。
- 它不是 GMR/AMASS Silver，不是实机真值，也不能证明 Any2Any；用途只是隔离“训练管线能否跟踪目标本体原生可执行 reference”。
- contact 是官方模型 active-sole 几何估计，**不是实机 GRF、COP、wrench 或足底力真值**。

## 31DoF → WBT29 合同

- 源 joint-name 集与官方 31DoF 完全一致，先按名字重排到 MuJoCo 31 顺序；WBT29 保持官方顺序。
- 两个头关节不进入 WBT：导出时锁到 model nominal，`dof_vel=0`；其余29关节逐样本保持 actual q/dq。
- MotionLib 存储仍为官方 MuJoCo 31 顺序；`pose_aa` root来自actual root quaternion，其余由轴×实际关节角构造，SMPL joints 明确置零。

## Audit

- 时间：3000 帧，59.981s；dt p99 误差 `0.002993s`。
- 连续性：joint-step `0.0958/0.4090rad` (p95/max)；dq 一致性 p95 `0.550rad/s`。
- limits：最大软越界 `0.0359rad`，样本比例 `0.00119`。
- recorded survival：root-z min `0.511m`，tilt max `34.12°`。
- active sole：最低 signed distance `-0.0073m`；DS/SS/flight `0.807/0.192/0.001`；DS→SS→DS `44` 次。
- command response（仅证据）：valid `1.000`，q-command p95 `0.462rad`，command Δ p99 `0.205rad`。

## 时间块与泄漏边界

| split | clips | frame blocks | duration | SS | DS | flight | cycles |
|---|---:|---|---:|---:|---:|---:|---:|
| train | 4 | [[0, 400], [400, 800], [800, 1200], [1200, 1600]] | 32.0s | 0.165 | 0.834 | 0.001 | 21 |
| held_out | 3 | [[1800, 2200], [2200, 2600], [2600, 3000]] | 24.0s | 0.198 | 0.800 | 0.002 | 19 |

- train `[0,1600)`，embargo `[1600,1800)`，held-out `[1800,3000)`；clip固定400帧、无重叠。随机相邻帧切分被禁止。

## 裁决

- 结果：official native actual-state trace通过完整性、动力学记录一致性、active-sole接触周期和时间块泄漏门，已导出独立Gold sanity seed。
- 结论：该资产可用于隔离X2原生可执行reference的训练/评估pipeline sanity；它不增加GMR Silver数量，也不证明跨本体迁移。
- 下一步：下一阶段只做pipeline tracking sanity对照，并严格保持held-out时间块；不得将train/held-out合并或随机切帧。
