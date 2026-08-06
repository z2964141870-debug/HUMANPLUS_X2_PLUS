# DC-PEFT Stage152-B 冻结基线

日期：2026-07-28

## 固定对象

- checkpoint：`model_step_000200.pt`
- SHA-256：`b73c345995c5d468c18d223de96b29fff4cd4866e4d56bb6d5296c4a68540679`
- 动作：4 条官方 X2 short-pulse forward walk（含镜像）
- seed：0
- 每域：4 env × 260 steps
- 评估：固定首帧、deterministic rollout、同一后处理门禁

## 可重复性

R1 与 R2 在 16 个动作-域组合上的以下连续指标逐项完全一致：

- reference displacement
- robot displacement
- along-reference progress
- progress ratio
- direction cosine
- root XY RMSE
- max foot error

最大绝对差：`0.0`，通过 `1e-6` 硬门。

## 四域冻结结果

| 域 | R1 | R2 | 稳定 | 主要失败 |
|---|---:|---:|---:|---|
| ideal | 0/4 | 0/4 | 4/4 | 位移普遍过冲、接触周期不稳 |
| nominal filter | 1/4 | 1/4 | 4/4 | 3 条仍明显过冲或足端超门 |
| nominal + delay | 0/4 | 0/4 | 4/4 | 过冲最大，镜像 B 出现反向位移 |
| nominal + delay + noise | 0/4 | 0/4 | 4/4 | 方向、接触和足端误差混合失败 |

## 结论

Stage152-B 并非合格步态策略，但它的失败是完全可复现的，因此可以作为 DC-PEFT 的冻结因果基线。后续成功标准不是仅提高训练 reward，而是在相同四域门禁上同时提高 progress、接触时序和足端指标，并保持上肢不退化。

说明：IsaacLab rollout 使用历史 Stage152 面板的宽松 termination，以取得全时序 trace；最终 pass/fail 由独立 auditor 判定。rollout 中记录的 reward 不参与确定性 pre-update actor action，也不应被解释为 Stage152 原训练 reward。
