# WBT Phase57 — PEFT-protected upper-robust lower live zero

## 假设

Phase56 的失败主要来自全量 dense actor/critic 在一次 PPO update 中产生巨大漂移。将 Stage219 dense actor、dense critic 和 action std 全冻结，仅在标准93D actor/critic的下层相关路径挂零输出 LoRA，应该在初始化时完全保留 source，同时为未来的小幅鲁棒适配提供受限参数子空间。

## 干预

- source：Stage219 `model_2600.pt`，SHA256 `abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb`。
- actor LoRA：`actor.0/2/4/6`；因为该actor只输出lower12+waist3，这四层全部属于腿腰动作路径。
- critic LoRA：独立的 `critic.0/2/4/6`，不允许critic dense更新。
- rank=4，alpha=4；`lora_A` Kaiming初始化、`lora_B`精确零，因此初始residual精确为0。
- 预注册未来学习率 `5e-5`：Phase56在`1e-3`下固定batch KL mean=5.696，因此只做一次20倍降幅，不扫学习率。
- 继续使用Phase56偶/奇配对：A/B各24 ideal + 8 response actuator env。

本阶段未创建optimizer、未执行环境control step、未训练、未生成checkpoint。

## 对照

同一 live 64×93 observation 与其翻转batch，比较原始Stage219与零初始化LoRA candidate的deterministic action/value；同时核验dense/std canonical hash、可训练参数名与15/29/31边界。

## 结果

- live observation：policy/critic均`64×93`；policy action=`15D`。
- 边界严格成立：`body29 = lower12+waist3(15) + external upper14`；`sim31 = body29 + nominal head2`。
- A/B domain计数：none=`24 ideal + 8 response`，bounded=`24 ideal + 8 response`。
- actor LoRA scope：`0/2/4/6`；critic独立LoRA scope：`0/2/4/6`。
- trainable tensors共16个，全部为`lora_A/lora_B`；trainable parameters=`9,000`，占含LoRA总参数`5.695%`。
- dense/std hash注入前后完全一致。
- 两个固定batch：action max-abs=`0.0`，value max-abs=`0.0`，全部finite；初始化source-retention KL=`0`。
- optimizer constructed=false；optimizer steps=0；environment control steps=0；checkpoint created=false。

## 结论

**PASS_LIVE_ZERO_UPDATE_ONLY**。

现有基础设施可以干净地把LoRA接到标准Stage219 `93D→15D` BASE，不需要复用WBT29/SONIC decoder的复杂注入器。Phase56暴露的dense一步漂移风险在结构上被隔离：dense actor/critic/std均不再可训练，初始action/value逐元素与source完全一致。

这仍不证明一次LoRA update会改善B，也不证明鲁棒性；本阶段没有物理rollout或optimizer。

## 下一步

停止并等待授权。若进入唯一1-update，必须fresh Stage219 + 当前rank4合同 + lr=`5e-5`，继续用Phase56 A/B前后评估与KL hard stop；结果无论通过或失败都先回报，不自动5-update。

证据：`reports/retarget/x2_upper_robust_lower_lora_live_zero_phase57.json`。
