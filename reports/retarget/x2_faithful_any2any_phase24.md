# X2 Faithful Any2Any Entrypoint — Phase24

- 裁决：**B3_B4_IMPLEMENTATION_READY_B5_FAIL_CLOSED**。
- 仅静态审计、Gold hook选择和CPU decoder/critic forward；没有Isaac physics、optimizer、PPO update或网络保存。
- B5尚未冻结，因此live zero-update由launcher以exit 42失败关闭。

## 假设

B3/B4可以先作为不可变训练合同完成，而不把未冻结的目标物理资产默认为正确。SONIC论文范围与Figure7 S7必须保留为两条独立对照。

## 干预与对照

- 干预：锁定source PPO、source3/source14 reward语义、source-equivalent DR、冻结std和Gold split hook。
- 对照A：SONIC actor dynamics decoder + critic。
- 对照B：Figure7 exact S7，actor input只开放proprio列，critic input/output冻结。

## 结果

- source/config checks：`{'source_checkpoint_hash': True, 'source_config_hash': True, 'ppo_exact': True, 'source3_exact': True, 'source14_exact': True, 'reward_terms_exact': True, 'source_equivalent_dr_exact': True, 'std_load_true_frozen': True}`。
- train keys：`['official_native_dance_train_000', 'official_native_dance_train_001', 'official_native_dance_train_002', 'official_native_dance_train_003']`；held keys：`['official_native_dance_held_out_000', 'official_native_dance_held_out_001', 'official_native_dance_held_out_002']`。
- A trainable：250728，占adapted dense `0.011561`；CPU checks `{'action_shape_29': True, 'value_shape_1': True, 'zero_B_action_exact': True, 'zero_B_value_exact': True, 'std_loaded_exact': True, 'std_frozen': True, 'only_lora_A_B_trainable': True, 'all_outputs_finite': True}`。
- B trainable：217080，占adapted dense `0.010010`；proprio mask `{'module.0': {'selected_columns': [64, 994], 'selected_count': 930, 'input_width': 994}}`；CPU checks `{'action_shape_29': True, 'value_shape_1': True, 'zero_B_action_exact': True, 'zero_B_value_exact': True, 'std_loaded_exact': True, 'std_frozen': True, 'only_lora_A_B_trainable': True, 'all_outputs_finite': True}`。
- 两条manifest的完整trainable names/shapes/masks在同名JSON中；std、reference encoders、kinematic decoder、FSQ和其他dense均声明冻结。

## 结论

- 结果：Faithful config, split-explicit entrypoints and two independent LoRA manifests pass static/CPU forward checks.
- 结论：B3/B4 are implementation-ready; this is not a live zero-update pass because B5 remains deliberately unresolved.
- 下一步：Freeze B5 target physics provenance, then wire the same immutable config into the preregistered live zero-update gate.
