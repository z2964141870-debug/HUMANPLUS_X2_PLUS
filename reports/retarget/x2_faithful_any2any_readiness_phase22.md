# X2 Faithful Any2Any Training Readiness — Phase22

- 裁决：**BLOCKED_BEFORE_ZERO_UPDATE**。
- 本阶段纯只读：没有训练、physics、网络修改、BASE或真机。
- Phase21 replay失败不构成Any2Any训练否证；这里检查的是训练合同是否忠实、可证伪。

## 总结

离线资产已经足以搭建sanity：SONIC checkpoint、Gold train/held-out、4秒embargo、MotionLib round-trip和31↔29名称映射均已冻结。当前仍不能启动优化，因为live训练环境仍是31维扩展边界，Gold state adapter未接入训练，且没有一份锁死PPO/reward/DR和LoRA范围的faithful配置。

## 冻结哈希

- checkpoint `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`
- source obs contract `250a2b8852ce9a9731022892ee96695ad12d52c0d2de4d3bf55e3876c21cdb61`
- source action contract `2cf4fc1742d25f123a06bef9f87ba755befc5654949956094d588d5c3e14d5cc`
- X2 31/29 alignment `d77a235a18637f84124a607fdd8e9f7962877fd860fd260e84b3882bf9d6960e`
- Gold train/held `644dc7534b63a7831bfe4156941b01508003f2834d2ccdac7b227369bad8ef8b` / `45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc`

## READY

- `R1_SOURCE_FROZEN`：Any2Any paper, SONIC last.pt and source config hashes match preregistration; checkpoint exposes 29-row g1_dyn and 1645-D critic.
- `R2_DATA_SPLIT_FROZEN`：Gold train=4x400, held-out=3x400 at 50Hz with a disjoint 200-frame/4.0s embargo and immutable hashes.
- `R3_MOTIONLIB_ROUNDTRIP`：Phase11 native loader + minimal state adapter passed q/root/FK/29-head-lock round-trip.
- `R4_OFFLINE_31_29_ALIGNMENT`：official31 partitions exactly into WBT29 + head2; WBT29 and G1 use the same named set and have an explicit bijective permutation.
- `R5_OPTIMIZATION_ENTRYPOINTS`：LoRA layer selection/masking, PPO, reward, motion-file and DR override entrypoints exist in local code.

## BLOCKER

- `B1_RUNTIME_29_ALIGNMENT_NOT_INSTANTIATED`：current X2 runtime/training expands policy observations/actions to 31 (Stage152 action_count=31) and masks head rows; it does not first scatter/gather through the frozen G1 29-D semantic space. 需要：dedicated WBT29 action term and observation/history adapter using the frozen permutations; head must be absent from policy and held nominal in the simulator.
- `B2_GOLD_STATE_ADAPTER_NOT_IN_TRAIN_PATH`：Phase11 state adapter is standalone audit code; current training config still points to receiver_safe_clean and does not prove actual dq/root velocity/model-contact restoration for Gold train clips. 需要：minimal immutable MotionLib hook for Gold train/eval, with pose/root/FK unchanged and split-specific paths.
- `B3_FAITHFUL_CONFIG_NOT_LOCKED`：generic launcher defaults are {'PPO_EPOCHS': '1', 'NUM_MINI_BATCHES': '1', 'TRAIN_BOUNDARY_KEYS': '[std]', 'TRACKED_BODY_SET': 'x2_full33', 'REWARD_POINT_MODE': 'x2_5point', 'RIGID_BODY_MASS_RANGE': '[0.8,2.5]', 'ACTUATOR_GAIN_RANDOMIZATION': 'off'}; source requires PPO epochs=5, minibatches=4, std frozen, source3/source14 reward semantics and source-equivalent DR. 需要：new preregistered faithful config/launcher; do not reuse mutable generic defaults.
- `B4_EXACT_S7_SCOPE_NOT_INSTANTIATED`：Stage152 adapts all g1_dyn rows/columns plus critic input/output; exact Figure7 S7 needs proprio-only input columns, actor hidden backbone/output and critic hidden backbone only. 需要：instantiate the exact named module/mask list and record trainable/frozen tensors; keep SONIC-specific decoder+critic baseline as a separate primary control.
- `B5_TARGET_PHYSICS_PROVENANCE_NOT_FROZEN`：current IsaacLab X2 config uses locally reconstructed URDF/collision/PD profiles, while Gold provenance is official AimDK v1 MuJoCo x2.xml/control; no morphology-equivalent Isaac contract hash is locked. 需要：freeze a declared IsaacLab target asset/PD/collision contract and state exactly how it corresponds to official v1; official MuJoCo remains held-out sim-to-sim, not training truth by implication.
- `B6_ZERO_UPDATE_NOT_EXECUTED`：Phase11 validates data ingestion, not policy initialization equivalence; no live 29-D aligned policy has produced identical zero-LoRA actions/tokens/value on a fixed batch. 需要：pass the preregistered zero-update gate below before any optimizer step.

## S7与当前实现

- 论文SONIC实验明确写的是 actor dynamics decoder + critic；Figure 7 S7来自Oli-WBT→Luna，二者必须作为两条独立对照，不可混称。
- 当前Stage152-B覆盖g1_dyn全部7层和critic全部7层，第一层没有proprio-only列mask，critic input/output也被训练，因此不是严格S7。
- 当前MLP的严格S7翻译：actor module.0只开放`proprioception`列；module.2/4/6/8/10作为backbone；module.12作为action output；critic只开放module.2/4/6/8/10。

## 数据隔离

- train ranges `[[0, 400], [400, 800], [800, 1200], [1200, 1600]]`，held-out ranges `[[1800, 2200], [2200, 2600], [2600, 3000]]`。
- embargo `200`帧 / `4.0s`，无range/key重叠。
- held-out不得进入optimizer、阈值调整、early stopping或checkpoint选择；本单条官方舞蹈只能做pipeline sanity，不能冒充Any2Any训练语料规模。

## Zero-update门（必须先过）

- checkpoint/config/paper/data/contract hashes equal this report
- runtime policy action dimension=29; policy/critic term order and dimensions equal frozen contracts
- target29->G1->target29 name round-trip is exact; head never enters obs/action/history
- MotionLib train sampler keys exactly four train keys; held-out and embargo keys absent from every optimizer/callback path
- LoRA B=0 gives target-aligned action mean and reference token equal to frozen source-space forward: max_abs<=1e-6
- std equals source and is frozen; all dense checkpoint tensors are unchanged by hash
- trainable names exactly equal selected SONIC-specific or S7 manifest; all other requires_grad=false
- critic running statistics are loaded by mapped semantics and are not silently reinitialized
- 精确工作量：`{'env_steps': 0, 'optimizer_steps': 0, 'fixed_forward_batches': 2}`；预算：one Isaac/model startup plus two forwards; budget <=10 min wall and no checkpoint output beyond a small manifest。

## 1-update门（zero通过后才允许）

- all rollout observations/actions/rewards/advantages/logprobs/losses/gradients finite
- 100% intended LoRA groups receive finite nonzero gradients; frozen dense/reference/FSQ/std tensors receive no gradient and keep exact hashes
- post-update effective LoRA delta is finite and nonzero; action dimension remains 29 and no head row appears
- approx KL finite and <=0.02 (2x source desired_kl); no optimizer-state or checkpoint reload mismatch
- train sampler/callback audit contains only four train keys; held-out metrics are computed only after update and never select/stop this single update
- post-save reload reproduces the same fixed-batch action mean within max_abs<=1e-6
- 精确工作量：`{'envs': 64, 'rollout_steps_per_env': 24, 'transitions': 1536, 'ppo_epochs': 5, 'mini_batches_per_epoch': 4, 'optimizer_minibatch_steps': 20}`；预算：budget <=20 min wall, <=1 candidate checkpoint; measure actual peak VRAM/wall time rather than treating estimate as evidence。
- 1 update只证明梯度/保存/隔离合同成立，不证明性能提升。

## 结论

- 结果：Local assets and offline data/alignment are ready, but the live faithful 29-D runtime/config/S7 contracts have not been instantiated.
- 结论：Do not train yet. Phase21 replay failure is outside this audit and does not reject Any2Any; readiness is blocked by implementation contracts, not by a demonstrated learning failure.
- 下一步：Implement only B1-B5, then run the zero-update gate. A one-update smoke is forbidden until zero-update passes.
