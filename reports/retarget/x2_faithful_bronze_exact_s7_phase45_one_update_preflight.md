# X2 Phase45 one-update live preflight

- 裁决：**ONE_UPDATE_BLOCKED_LIVE_WBT29_EXACT_S7_NOT_WIRED**。
- 已完成的Bronze split与exact-S7 B=0仍有效；阻塞发生在live PPO接线，而非数据或学习结果。
- 未创建Isaac环境、未走physics step、未创建optimizer、未生成checkpoint。

## 假设

若Phase23–26真的已进入live trainer，则Stage152入口应呈现29维source-semantic action/observation/reference，并只给exact-S7指定层梯度。

## 干预与对照

只读比较Phase45冻结合同与当前Stage152 launcher、action cfg、policy/critic observation及reference observation；没有运行训练。

## 结果

- `LIVE_ACTION_OUTPUT_IS_31_NOT_WBT29`：The action term selects [.*], so the live action manager and actor output are 31-D including both head rows; Phase45 exact-S7 requires a 29-D source-semantic output followed by explicit scatter and nominal head lock.
- `LIVE_POLICY_CRITIC_HISTORY_IS_31_NOT_WBT29`：policy/critic joint_pos, joint_vel and actions terms have no explicit 29-joint name/order selector; the offline WBT29 gather/permutation is not called by the live environment.
- `LIVE_REFERENCE_ENCODER_INPUT_IS_31_JOINT`：command_multi_future_nonflat calls the raw all-joint command. The live encoder therefore uses the X2-expanded 31-joint boundary rather than the frozen source 29-joint 640-D contract.
- `LIVE_CRITIC_LORA_SCOPE_IS_NOT_EXACT_S7`：run_dcpeft_stage152.sh hardcodes critic_lora_prefixes=[critic_module], which includes critic module.0 and module.12; exact-S7 freezes both and adapts only 2/4/6/8/10.
- `PHASE23_HOOK_NOT_LIVE_WIRED`：WBT29PolicyContract and ImmutableGoldMotionLibHook are exercised by CPU/static tools but are absent from the live Stage152 launcher/trainer path.

## 结论

- 结果：The Bronze data and offline B=0 gate are valid, but the only live PPO launcher does not implement the frozen WBT29/exact-S7 contract.
- 结论：Starting it would run a different 31-D Stage152 experiment and would violate the authorized fail-closed contract. No optimizer or physics was started.
- 下一步：Implement a dedicated opt-in live WBT29 observation/action/reference adapter plus exact critic layer prefixes, then run an env-initialization/no-update shape/hash probe before re-authorizing the same one-update pilot.
