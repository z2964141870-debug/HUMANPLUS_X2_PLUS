# X2 WBT29 + Native Gold Hook — Phase23

- 裁决：**B1_B2_IMPLEMENTATION_READY_CPU_PROBED**。
- 本阶段只有纯单测与 CPU MotionLib zero-step probe；未启动 Isaac physics、PPO、LoRA、optimizer 或真机。
- `model_contact` 是官方仿真碰撞几何标签，不是实机 GRF/COP/wrench。

## 假设

B1/B2 可以通过两个小型、可组合的边界实现：策略始终只看 G1 source-semantic 29DOF，模拟器31DOF中的头部保持显式 nominal；Gold train/held-out 由哈希和key分别锁定，状态adapter只恢复actual dq/root velocity/model contact。

## 干预

- 新增 name-driven WBT29 observation/history/action gather/scatter。
- 新增 immutable Gold split hook；train可供后续optimizer sampler，held-out硬标记不可进入optimizer。

## 对照

- 31→source29→target29/31 round-trip 与原始按名切片逐元素比较。
- MotionLib attach 前后逐元素比较 q、root pose 和全身 FK；actual dq/root velocity/contact 与源PKL比较。

## 结果

- B1 checks：`{'source29_dim': True, 'target_roundtrip_exact': True, 'official_scatter_nonhead_exact': True, 'official_scatter_head_nominal_exact': True, 'history_terms_exact': True, 'history_joint_pos_source_exact': True, 'history_joint_vel_source_exact': True, 'flat_policy_dim_930': True}`。
- train keys：`['official_native_dance_train_000', 'official_native_dance_train_001', 'official_native_dance_train_002', 'official_native_dance_train_003']`。
- held-out keys：`['official_native_dance_held_out_000', 'official_native_dance_held_out_001', 'official_native_dance_held_out_002']`。
- train/held CPU probes：`True` / `True`。
- policy history/action dims：`930` / `29`；head exclusion：`['head_yaw_joint', 'head_pitch_joint']`。

## 结论

- 结果：WBT29 gather/scatter/history and immutable train/held Gold hooks pass exact CPU zero-step contracts.
- 结论：B1/B2 are implementation-ready as isolated composable hooks; they are not yet wired by a faithful live train/eval config, and live Isaac/PPO integration remains deliberately unexecuted.
- 下一步：Freeze this manifest, then address Phase22 B3-B5 before the preregistered live zero-update gate.
