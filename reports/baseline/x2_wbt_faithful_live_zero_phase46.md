# WBT Phase46 — faithful live entrypoint / zero-update

## 假设

Phase23–25 的 WBT29、immutable MotionLib hook、exact-S7 与 B5 训练域合同，只有真正接入 Stage152 live entrypoint 并保持 source checkpoint 的观测/动作形状，才具备后续 1-update 的工程资格。

## 干预

- 新增 opt-in `FAITHFUL_WBT29=true` 路径；历史 Stage152 默认路径不变。
- policy action/history/proprioception 均使用冻结的 G1 29-joint 顺序；X2 两个头关节只存在于 31DoF articulation，默认值严格为零。
- critic 使用发布 checkpoint 的 14-body source semantic contract，因此 live input 保持 1645 维；没有通过复制或裁剪权重掩盖维度差异。
- actor exact-S7 注入 `0/2/4/6/8/10/12`，其中 module0 只允许 proprioception 列；critic 只注入 `2/4/6/8/10`，明确不含 module0/12。
- Bronze train 与 native-Gold held-out 使用不同的哈希锁定 MotionLib；held-out 三个 key 全量载入，避免 `max_unique_motions=1` 的随机抽样。

## 对照

每个 split 都从同一个 SONIC source checkpoint 构造 base 与零初始化 exact-S7 模型，在同一 reset observation 上做两个固定 forward batch；比较 actor action 与 critic value。未创建 optimizer，也未调用 `env.step`。

## 结果

| 项目 | Bronze train | native-Gold held-out |
|---|---:|---:|
| immutable sampler keys | 3/3 | 3/3 |
| policy obs | 930 | 930 |
| critic obs | 1645 | 1645 |
| tokenizer future | 10×58 | 10×58 |
| policy action | 29 | 29 |
| articulation/head | 31 / head2 nominal | 31 / head2 nominal |
| 两批 action B=0 max abs | 0.0 | 0.0 |
| 两批 value B=0 max abs | 0.0 | 0.0 |
| finite | PASS | PASS |

运行事实边界：共初始化两个 Isaac 环境、每个只 reset 一次以取得真实 observation；control steps=0、optimizer instances/steps=0、checkpoint created=0。环境初始化可能让模拟器完成内部启动，但本阶段没有执行策略控制物理步。

## 结论

Phase46 **PASS（仅 live zero-update）**。此前的 live blocker 已闭合：29↔31 边界、14-body source critic、Bronze/held-out split、Gold state adapter、exact-S7 layer scope 与 B=0 数值等价都已由真实 Stage152 runtime 验证。

这不证明参考动作可执行、X2 能稳定跟踪，也不等于完成 PPO。Bronze 仍只能称 kinematic Bronze；native Gold 只用于 pipeline/原能力回归，不是跨具身 held-out。

## 下一步

保持停止：不自动运行 1-update。若主任务授权，下一阶段只能执行 Phase45 已预注册的唯一 1-update pilot，并在任何动态 lunge 方向、termination/survival、NaN/grad 或 native-Gold 原能力门失败时停止，不能自动进入 5-update。

## 证据

- `logs/x2_faithful_live_zero_phase46_train/phase46_live_zero_train.json`
- `logs/x2_faithful_live_zero_phase46_held_out/phase46_live_zero_held_out.json`
- `reports/x2_wbt_faithful_live_zero_phase46.json`
- `tests/test_x2_faithful_live_phase46.py`
