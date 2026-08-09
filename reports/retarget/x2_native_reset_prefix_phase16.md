# X2 Reset-Compatible Prefix Phase16

- 裁决：**PHASE16_RESET_PREFIX_CONTRACT_REJECTED_NO_PHYSICS**。
- v2 mode-event 扩展本身成功；但预注册的完整4秒 actual-state prefix 不成立，因此未运行 prescribed/free physics。
- 本结果只否定这一次capture对Phase16假设的资格，不否定官方policy、X2动力学、Any2Any或reset-prefix思路。

## 假设

Phase15 的动态中段冷启动若是主要误差源，则从 JOINT_DEFAULT 后首个完整 actual q/dq/root/command context 开始，重放至少4秒稳定前缀，再进入RL段，应改善trackability。

## 干预 / 对照

- 唯一计划变量：Phase15动态中段冷启动 → reset-compatible JOINT_DEFAULT稳定前缀。
- recorder先ready；domain215隔离；只采一次；mode-event保存callback monotonic、buttons、header和推断mode。
- 合同停止门：从JOINT后首个完整snapshot到RL切换必须≥4.0s；否则不允许physics。

## Capture结果

- `1063` complete snapshots，`52912` command events；mode sequence为 `JOINT_DEFAULT → RL_DEFAULT`。
- JOINT receipt `0.936s`；RL receipt `6.077s`；表面间隔 `5.141s`。
- 但首个完整31DOF snapshot直到 `5.195s` 才出现；到RL仅余 `0.882s`，低于4秒。
- RL后完整snapshot覆盖 `20.378s`；该部分充足，但不能补回缺失的前缀reference。
- controller日志确认先JOINT再RL；Joy header仍未填充，因此mode时间是subscriber receipt，不是controller内部生效时间。

## Gate

- `npz_hash_matches_manifest`: `True`
- `isolated_domain_215`: `True`
- `mode_event_sequence_exact`: `True`
- `mode_event_receipt_monotonic`: `True`
- `mode_event_buttons_exact`: `True`
- `command_global_index_exact`: `True`
- `command_receipt_monotonic`: `True`
- `snapshot_receipt_monotonic`: `True`
- `controller_acknowledged_sequence`: `True`
- `raw_mode_interval_at_least_4s`: `True`
- `first_complete_snapshot_after_joint_exists`: `True`
- `complete_reference_prefix_at_least_4s`: `False`
- `rl_segment_at_least_20s`: `True`

- 失败项：`['complete_reference_prefix_at_least_4s']`。
- prescribed executed：`False`；free-root executed：`False`。

## 结论

- 结果：JOINT→RL mode receipts span 5.141s, but the first complete 31DOF snapshot appears at 5.195s, leaving only 0.882s of scoreable prefix.
- 结论：Mode-event capture succeeded, but this capture cannot test the reset-prefix hypothesis; missing complete actual-state reference is a capture-contract failure, not a dynamics or Any2Any result.
- 下一步：Stop Phase16 without physics. A future separately authorized capture must wait until complete snapshots are available and then hold JOINT_DEFAULT for four additional seconds before RL_DEFAULT.
