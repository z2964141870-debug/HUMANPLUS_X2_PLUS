# WBT Phase59 — upper-robust LoRA 最多5-update early-stop pilot

## 假设

Phase58 的单次LoRA update通过了局部趋势门。若该改善不是偶然波动，那么从完全相同的Stage219零LoRA source fresh start，逐update训练并即时评估，应至少复现第1次B改善，随后才有资格继续到最多5次。

## 干预

- 严格复用Phase58：rank=4、alpha=4、fixed LR=`5e-5`、seed=42。
- 每update：64×24=`1536 transitions`，5 epochs×4 minibatch=`20 optimizer steps`。
- A/B偶奇配对，各24 ideal + 8 response。
- dense actor/critic/std冻结，只训练9,000个LoRA参数。
- 每次update后立即跑固定4秒A/B面板；任何硬门失败即停止。
- 最佳权重按`A robust score + B robust score`选，不默认最后。

## 对照

fresh source与Phase58 source的所有模型tensor逐元素最大差为`0.0`，因此Phase59不是从错误或不同的初始化开始。Phase58曾在同一预算下得到A/B score分别`+0.02628/+0.02148`。

## 结果

Phase59在**第1个update**即触发早停，没有运行update 2–5。

### 数值合同

- loss有限：value=`0.09599`，surrogate=`-0.002484`，entropy=`0.25264`。
- dense与std hash均保持不变；20 optimizer steps正确。
- cumulative KL mean=`0.000253`、max=`0.000297`；action/value drift=`0.002092/0.02932`，均远低于门限。
- survival、termination、root、action和A保护的独立硬门均通过。

### 失败门

唯一失败项是 **B_robust_score_improved=false**：

| 指标 | source | update1 |
|---|---:|---:|
| B survival | 4.000 s | 4.000 s |
| B termination | 0.000 | 0.000 |
| B velocity RMSE | 0.10437 | 0.10668 |
| B lateral abs | 0.06802 m | 0.07836 m |
| B yaw abs | 0.15394 rad | 0.15257 rad |
| B robust score | 3.26413 | 3.25427 |
| B score delta | — | **-0.00986** |

A score同样由3.27604降到3.24889，但各项仍在保护容差内。joint A+B score由`6.54017`降至`6.50316`。

## 结论

**EARLY_STOP_UPDATE1_NO_PASS_SOURCE_REMAINS_BEST**。

Phase58的微小正向结果没有在完全相同source state、配置、seed和预算下稳定复现。由于source state逐元素完全相同，而两次update方向相反，最诚实的判断是：当前提升量级小于训练/仿真运行波动，尚不能作为可靠学习信号。

这不是LoRA数值爆炸：KL、dense保护、生存都健康；失败发生在“B必须真正改善”的核心命题。因此按照预注册规则，整个upper-robust训练路线在此终止，不扫LR/rank，也不靠更多update把负结果平均掉。

## 权重清理

只保留2份必要证据：

- `source_stage219_zero_lora.pt`：同时是best-update0。
- `rejected_last_update01.pt`：第1次失败候选。

没有update2–5或冗余中间checkpoint。

## 下一步

本路线没有自动下一训练实验。若将来重新开启，应先增加多seed/paired rollout统计能力，证明单update效应超过运行方差；不能直接延长训练或继续调整LoRA超参数。

机器证据：`reports/retarget/x2_upper_robust_lower_lora_five_update_phase59.json`。
