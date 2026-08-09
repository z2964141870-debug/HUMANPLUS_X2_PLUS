# WBT Phase58 — PEFT-protected upper-robust lower 唯一 1-update

## 假设

Phase56 的一次dense更新因策略KL过大而同时摧毁A/B。若严格冻结Stage219 dense actor/critic/std，只训练Phase57的rank-4 LoRA，并把固定学习率降至`5e-5`，则一次update应能在保护A的同时，让B的上肢扰动鲁棒指标出现正确方向。

## 干预

- fresh Stage219 + Phase57零输出LoRA；rank=4、alpha=4。
- trainable仅16个LoRA tensor、9,000参数：actor与critic的`0/2/4/6.lora_A/B`。
- dense actor/critic、std与其余环境合同全部冻结。
- fixed LR=`5e-5`，无adaptive schedule，不扫LR/rank。
- seed=42；64 env×24 step=`1536 transitions`；5 epochs×4 minibatch=`20 optimizer steps`。
- 偶/奇配对：A none与B bounded各自均为24 ideal + 8 response actuator env。
- 仅保留source/final两份权重，没有中间checkpoint。

## 对照

update前后分别运行同一个4秒评估：命令`[0.35, 0, 0]`，无外部push，统计A none、B bounded及其ideal/response子组。所有门在optimizer前预注册。

## 结果

### 训练合同与source retention

- optimizer steps=`20`，仅两份权重；所有loss、grad与输出finite。
- loss：value=`0.09585`，surrogate=`-0.002802`，entropy=`0.25264`。
- dense hash前后相同；std hash前后相同；trainable names全部为LoRA。
- fixed-batch action drift max=`0.002372`，value drift max=`0.02872`。
- KL mean=`0.000158`，max=`0.000200`，远低于预注册`0.02/0.20`；相比Phase56 dense更新KL mean=`5.696`，降低约36,000倍。

### A：无上肢扰动保护

| 指标 | update前 | update后 | 判断 |
|---|---:|---:|---|
| survival | 4.000 s | 4.000 s | 保持 |
| termination | 0.000 | 0.000 | 保持 |
| velocity RMSE | 0.10505 | 0.10576 | 容差内 |
| yaw-rate RMSE | 0.37021 | 0.36063 | 改善 |
| lateral abs | 0.08396 m | 0.07146 m | 改善 |
| yaw abs | 0.13837 rad | 0.13345 rad | 改善 |
| action abs | 0.69746 | 0.69407 | 略降 |
| robust score | 3.27604 | 3.30233 | `+0.02628` |

### B：有界上肢扰动

| 指标 | update前 | update后 | 判断 |
|---|---:|---:|---|
| survival | 4.000 s | 4.000 s | 保持 |
| termination | 0.000 | 0.000 | 保持 |
| velocity RMSE | 0.10437 | 0.10448 | 基本持平 |
| yaw-rate RMSE | 0.38300 | 0.37765 | 改善 |
| lateral abs | 0.06802 m | 0.05783 m | 改善 |
| yaw abs | 0.15394 rad | 0.14792 rad | 改善 |
| upper tracking RMSE | 0.03941 rad | 0.03913 rad | 略改善 |
| action delta | 0.26535 | 0.26502 | 略改善 |
| robust score | 3.26413 | 3.28561 | `+0.02148` |

全部预注册硬门通过，包括finite、精确20步、两权重、dense/std冻结、KL、A保护与B改善。

## 结论

**PASS_LORA_ONE_UPDATE_TREND_GATE_STOP**。

这是相比Phase56的明确方法学推进：同样的1536 transitions与20 minibatch steps，全dense更新会在一步内使A/B全部倒地，而受保护LoRA更新把KL限制在`1.58e-4`，维持A/B 4秒零终止，并让B的横向、航向和综合分数小幅改善。

但提升仍很小，且仅是单一速度、4秒、一次update的局部趋势；不等于全身遥操作已解决，也不证明继续训练必然单调改善。

## 下一步

按授权在此停止，**不自动5-update**。如果后续批准累计小试，应从原始source fresh start并继续逐update门控，而不能从Phase56失败dense权重续训；任何KL或A保护门失败立即终止。

证据：

- `reports/retarget/x2_upper_robust_lower_lora_one_update_phase58.json`
- `reports/retarget/x2_upper_robust_lower_lora_phase58_pre.json`
- `reports/retarget/x2_upper_robust_lower_lora_phase58_train.json`
- `reports/retarget/x2_upper_robust_lower_lora_phase58_post.json`
