# WBT Phase47：faithful exact-S7 唯一 1-update pilot

## 裁决

**唯一一次 1-update 数值与即时非退化门通过；按约停止，5-update 仍锁定。**

这次结果证明的是：Phase46 接好的 live WBT29 / exact-S7 / Bronze-only 管线，确实能够完成一次真实 PPO update，而且没有立即破坏动态 lunge 或 native Gold 原能力。它不证明动作已经可用，也不证明长训会持续改善。

## 假设

在冻结的数据、关节合同、exact-S7 注入范围与物理配置下，一次小更新应当满足：loss/gradient 有限；动态 lunge 的 survival/termination 不退化，至少一项跟踪误差改善且没有误差恶化超过 10%；native Gold 不进入 optimizer，并且更新前后原能力不过门槛退化。

## 干预

- seed `0`，`64 env × 24 step = 1536 transitions`；
- `5 PPO epochs × 4 minibatches = 20 optimizer minibatch steps`；
- optimizer 只见 `AMASS-STAND-001`、`AMASS-UPPER-001`、`PHUMA-LUNGE-R-001` 三条 **kinematic Bronze**；
- native Gold 三条只做更新前后零优化回归；
- actor LoRA：`0/2/4/6/8/10/12`，module0 只开放 proprio 列；critic LoRA：`2/4/6/8/10`；
- 只保存 `source_B0.pt` 和 `last.pt`，没有中间 checkpoint。

## 对照

- source：SONIC/G1 原 checkpoint，经冻结的 WBT29→X2 合同加载；
- final：唯一一次 update 后的 `last.pt`；
- Bronze train 与 native Gold 使用同一 deterministic regression runner，motion id、seed、初态与 horizon 前后一致；
- native Gold 是 pipeline/source-ability 回归，不作为跨具身 held-out 性能证据。

## 结果

### 1. 更新本身

- 真实完成：`1 outer update / 20 optimizer steps / 1536 transitions`；
- `approx KL = 0.01670`，高于 desired `0.01`，但低于 hard stop `0.02`；
- policy loss `-0.01456`，value loss `12.9892`；actor/critic preclip grad `3.6859 / 20.0949`，均有限；
- frozen parameter hash 更新前后完全一致；全部 exact-S7 LoRA 参数发生非零变化；
- native Gold optimizer sample 数为 `0`。

### 2. 动态 lunge 即时门

- survival：`0.22 s → 0.22 s`；termination：`1.0 → 1.0`；没有退化；
- body-position error 改善 `0.184%`；其余误差变化均小于 `0.7%`，最差远低于 `10%`；
- 因而严格满足预注册的“至少一项改善、无一项恶化超过 10%”方向门。

但要诚实强调：**lunge 仍在 0.22 秒终止**。这只是“一次更新没有把它训坏”，不是已经学会动态 lunge。

### 3. native Gold 原能力门

- 三条 survival：`0.72/0.39/0.78 s → 0.72/0.38/0.78 s`；最坏相对下降 `2.56% < 5%`；
- termination rate 没有增加；单项 tracking 最坏恶化 `1.19% < 10%`；六项 tracking 的跨动作均值中五项改善，仅 joint velocity 恶化 `0.215%`；
- 固定初始 action drift：RMSE `0.00231`，max `0.00634`（source action RMS `1.1231`）；
- 固定初始 value drift：RMSE `0.03710`，max `0.04694`（source value RMS `6.2596`）。

### 4. 未闭合的精确门

保存后的 `last.pt` 已被两次 post regression 成功重新加载，输出有限；但 trainer 没有保存与其对应的“保存前同一输入 in-memory action”。因此 Phase45 预注册的 `post_save_reload_action_max_abs <= 1e-6` **不能由现有证据直接计算**。不能把“checkpoint 能加载”冒充“保存前后逐元素等价”。

## 结论

这轮是有效推进：此前只证明静态接线和 B=0 等价，现在首次证明 faithful exact-S7 live optimizer 在真实 Isaac 环境中可以完成一次受控更新，并通过 lunge/native-Gold 即时非退化门。

但它仍只是 optimizer sanity，不是 WBT 成功。由于精确 post-save reload 门缺少 pre-save tensor，而且动态 lunge 仍只有 `0.22 s` 生存，**5-update 不授权，也没有自动继续训练**。

## 下一步

按任务要求在此停止并汇报。若后续另行授权，先给 trainer 增加 pre-save 固定输入 action capture，闭合 `1e-6` reload 门，再决定是否值得进入最多 5-update；不能因为本轮 loss 有限就自动放长训练。

主要证据：

- `logs/x2_faithful_exact_s7_one_update_phase47/phase47_one_update_runtime.json`
- `logs/x2_faithful_phase47_regression_pre_train/phase47_regression_pre_train.json`
- `logs/x2_faithful_phase47_regression_post_train/phase47_regression_post_train.json`
- `logs/x2_faithful_phase47_regression_pre_held_out/phase47_regression_pre_held_out.json`
- `logs/x2_faithful_phase47_regression_post_held_out/phase47_regression_post_held_out.json`
