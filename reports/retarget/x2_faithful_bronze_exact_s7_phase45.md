# X2 Bronze-only Faithful exact-S7 — Phase45

- 裁决：**BRONZE_EXACT_S7_ZERO_PASSED_ONE_UPDATE_PREREGISTERED_NOT_RUN**。
- 方法学更正：Any2Any的target PPO可使用目标本体GMR运动学reference；动态Silver不是论文先验硬门。
- 真实性边界：3条训练reference只标为`kinematic Bronze`；绝不称Silver、动力学真值、实机GRF/COP或free-root稳定证据。

## 假设

Phase23–26的WBT29、B1–B5和exact-S7若能无改写接收Bronze sampler，则B=0应逐元素复现source，且Gold held-out不得进入optimizer。

## 干预

- train：AMASS stand、AMASS upper、PHUMA lunge三条Bronze；固定zero train batch使用唯一动态意图lunge。
- held-out：Phase10 native Gold只作pipeline/原能力回归，不能冒充cross-embodiment held-out。
- 只做2个CPU forward；Isaac环境、physics step、optimizer、PPO均为0。

## 对照与结果

- Bronze provenance：`{'selected_keys_exact': True, 'all_selected_are_bronze': True, 'none_selected_are_silver': True, 'evidence': {'AMASS-STAND-001': {'source': 'phase28_tier_report', 'tier': 'Bronze', 'bronze_pass': True, 'silver_pass': False, 'recommended_split': 'train_candidate', 'entry_sha256': '1203e1cc4d9196967fb20bb51688ce7c64012dfc7bf42d234b2b25110e260d78'}, 'AMASS-UPPER-001': {'source': 'phase28_tier_report', 'tier': 'Bronze', 'bronze_pass': True, 'silver_pass': False, 'recommended_split': 'train_candidate', 'entry_sha256': '9d54e4f44e7981fa48a2bda6d639d60348664d303a2cbfa5ee957b04aec6c786'}, 'PHUMA-LUNGE-R-001': {'source': 'phase36_tier_correction', 'historical_tier': 'Silver', 'tier': 'Bronze', 'bronze_pass': True, 'silver_pass': False, 'split': 'train_candidate', 'entry_sha256': '1ace093d6a0d22e26ffc281797f76465c45441a6bc1543427d9b305e1597cfee', 'historical_silver_revoked': True}}, 'phase36_silver_revocation_preserved': True}`
- B1–B5/hash guards：`{'all_frozen_hashes_exact': True, 'phase25_declared_train_domain_ready': True, 'phase25_files_exact': True, 'phase25_runtime_exact': True, 'phase26_B1_B5_zero_historical_pass': True, 'phase24_exact_s7_static_pass': True}`
- lunge train B=0：`{'input_finite': True, 'reference_token_max_abs': 0.0, 'action_source_max_abs': 0.0, 'action_target_max_abs': 0.0, 'action_official_nonhead_max_abs': 0.0, 'value_max_abs': 0.0, 'std_source_exact': True, 'head_nominal_exact': True, 'all_outputs_finite': True, 'token_sha256': '90bd158cb9f6f866a20c831ba3e9d39899769108954b95001af7448976460439', 'action_source_sha256': 'f0520006e689b9bde0fdba07a09886d21568fb77ec389c6aa6d17f1754b0bd39', 'value_sha256': 'def491819b31202fc38810773b972109d10663feb82db5be741ee81804db94ff'}`
- Gold held B=0：`{'input_finite': True, 'reference_token_max_abs': 0.0, 'action_source_max_abs': 0.0, 'action_target_max_abs': 0.0, 'action_official_nonhead_max_abs': 0.0, 'value_max_abs': 0.0, 'std_source_exact': True, 'head_nominal_exact': True, 'all_outputs_finite': True, 'token_sha256': '6ff0cde40b03acb8b6a5402946d27cf50246a7d0827a3fef755686527e1b553b', 'action_source_sha256': '7716d196fe2f3e71a0d1120b66dd358ee9269321c94f9449f7047b7374d79ab6', 'value_sha256': 'f5f1cf4e18f795bdedd6a9a738b06f11164fc050a53cc9abb09b031b52a31a74'}`
- zero checks：`{'guards_pass': True, 'bronze_provenance_pass': True, 'bronze_sampler_keys_exact': True, 'dynamic_train_key_is_lunge': True, 'held_gold_optimizer_ineligible': True, 'held_gold_absent_from_train_sampler': True, 'fixed_forward_batches_exactly_two': True, 'zero_B_token_exact': True, 'zero_B_target_action_exact': True, 'zero_B_value_exact': True, 'head_nominal_exact': True, 'std_source_exact': True, 'all_finite': True, 'dense_checkpoint_hash_unchanged': True, 'exact_s7_trainable_names': True, 'no_optimizer_or_physics': True, 'one_update_not_executed': True, 'five_update_locked': True}`

## 结论

- 结果：Three explicitly Bronze GMR references are isolated to the train sampler; the dynamic lunge and native-Gold held batch both reproduce source token/action/value at exact-S7 B=0.
- 结论：The corrected Bronze-only numerical path is ready for a separately executed one-update sanity. This is not PPO performance evidence and does not unlock five updates.
- 下一步：Run exactly one preregistered 20-minibatch update only after review; evaluate lunge direction/termination and Gold original-ability regression. Five-update remains locked unless every additional gate passes.

## 1→5 update预注册边界

- 1-update是20个optimizer minibatch的数值sanity，不是性能证明；本阶段没有执行。
- 仅loss有限不能解锁5-update；必须lunge即时诊断无NaN、termination不恶化且tracking方向正确，同时原能力回归过门。
- held Gold永不进optimizer，且只代表pipeline/原能力；当前没有独立cross-embodiment held-out性能证据。
