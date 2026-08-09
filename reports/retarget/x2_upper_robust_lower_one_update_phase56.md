# WBT Phase56 — upper-robust lower 唯一 1-update 裁决

## 假设

在标准 Stage219 `93D → 15D` lower12+waist3 actor 中加入物理上肢目标扰动课程，允许 actor/critic 各做一次 PPO update，可能让 B（bounded upper）下的根部、航向和速度跟踪开始改善，同时保持 A（upper none）的原有能力。

## 干预

- fresh start：Stage219 `model_2600.pt`，原始 SHA256 `abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb`。
- seed=42；64 env；24 steps/env；共1536 transitions。
- PPO：5 epochs × 4 minibatches，共**精确20次 optimizer.step**；learning rate=1e-3，adaptive KL target=0.01。
- actor+critic dense参数可训练，`std`冻结。
- upper sampler：偶数env无上肢扰动、奇数env有界上肢扰动；两组各24 ideal + 8 response actuator env。
- 只保留两份实验权重：`source_stage219.pt` 与 `final_one_update.pt`；没有中间checkpoint。

## 对照

update前后均从相同seed和固定评估配置运行4.0s：速度命令 `[0.35, 0, 0]`，无外部push；分别统计 A none 与 B bounded，并保留ideal/response子组。预注册硬门在 optimizer 前写入 `x2_upper_robust_lower_one_update_phase56_prereg.json`。

## 结果

### 数值与合同

- loss全部有限：value=`0.72312`，surrogate=`0.24289`，entropy=`0.25264`。
- optimizer steps=`20`；权重数=`2`；`std` hash前后完全相同。
- source model hash仍为Phase55的 `c8727a...4830`，证明fresh start正确。
- 但固定batch策略漂移严重：action max-abs=`0.2220`，value max-abs=`0.7652`，KL mean=`5.6961`、max=`6.9618`，远超预注册 `0.02/0.20`。

### A：无上肢扰动

| 指标 | update前 | update后 |
|---|---:|---:|
| survival mean | 4.000 s | 1.685 s |
| termination rate | 0.000 | 1.000 |
| velocity tracking RMSE | 0.105 | 0.380 |
| yaw-rate tracking RMSE | 0.370 | 1.386 |
| lateral abs | 0.084 m | 0.060 m |
| yaw abs | 0.138 rad | 0.331 rad |
| robust score | 3.276 | -2.498 |

只有横向平均偏移没有越门，其余核心稳定性与跟踪门均失败。

### B：有界上肢扰动

| 指标 | update前 | update后 |
|---|---:|---:|
| survival mean | 4.000 s | 1.743 s |
| termination rate | 0.000 | 1.000 |
| velocity tracking RMSE | 0.104 | 0.384 |
| yaw-rate tracking RMSE | 0.383 | 1.339 |
| lateral abs | 0.068 m | 0.063 m |
| yaw abs | 0.154 rad | 0.313 rad |
| upper tracking RMSE | 0.0394 rad | 0.0389 rad |
| robust score | 3.264 | -2.383 |

上肢自身跟踪误差略降，但机器人全部提前终止，不能称为鲁棒性改善；B robust score变化为 `-5.647`。

## 结论

**FAIL_ONE_UPDATE_TREND_GATE_STOP**。

该结果不是“训练没有运行”或NaN，而是更明确的负结果：按原Stage219全actor+critic、1e-3、5×4 PPO合同，一次update就造成过大的policy KL，并同时摧毁A和B的生存/跟踪。它不能归因于upper与actuator域混杂，因为Phase56已经用偶/奇交错将两组都平衡为24 ideal + 8 response。

因此本分支严格停止，**禁止自动5-update**。`final_one_update.pt`仅保留为失败证据，不得标记best或用于部署。

## 下一步

本阶段不继续实验。若未来重新授权，唯一高信息增益变量应是“保留同一数据与A/B配对，但把全actor更新替换为带显式KL/trust-region约束的低幅值结构化下层适配”；不能直接把学习率/epoch/LoRA一起扫，也不能从本失败final续训。

证据：

- `reports/retarget/x2_upper_robust_lower_one_update_phase56.json`
- `reports/retarget/x2_upper_robust_lower_phase56_pre.json`
- `reports/retarget/x2_upper_robust_lower_phase56_train.json`
- `reports/retarget/x2_upper_robust_lower_phase56_post.json`
