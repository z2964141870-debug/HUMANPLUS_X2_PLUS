# X2 checkpoint 形状审计

日期：2026-08-15  
类型：只读 `torch.load` 审计，没有写回 checkpoint

## 结论

当前必须把两类模型分开：

1. X2-native lower-velocity 环境：静态探针得到 policy 86、critic 89；
2. Stage152-B SONIC universal-token/Any2Any checkpoint：actor `g1_dyn` 输入 1062、critic 输入 1745。

它们不是同一个 observation contract，不能用一个 adapter 配置或一个训练结果代替。

## Stage152-B 证据

外部 artifact：`x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/model_step_000200.pt`  
大小：195741434 bytes  
SHA256：`b73c345995c5d468c18d223de96b29fff4cd4866e4d56bb6d5296c4a68540679`

顶层 state keys：`policy_state_dict`、`value_state_dict`、`optimizer_state_dict`、`lr_scheduler_state_dict`、`env_state_dict`、`state`、`args`。

关键张量：

| 组件 | checkpoint tensor | 形状 |
|---|---|---:|
| actor | `actor_module.decoders.g1_dyn.module.0.base_layer.weight` | `(2048, 1062)` |
| actor | `actor_module.decoders.g1_dyn.module.0.lora_A` | `(8, 1062)` |
| actor | `actor_module.decoders.g1_dyn.module.12.base_layer.weight` | `(31, 512)` |
| critic | `critic_module.module.0.base_layer.weight` | `(2048, 1745)` |
| critic | `critic_module.module.0.lora_A` | `(8, 1745)` |
| critic | `critic_module.module.12.base_layer.weight` | `(1, 512)` |
| actor noise | `std` | `(31,)` |

## 接入决策

- 不能把六链 36H 维目标直接追加到 Stage152-B 的 86 维假想输入；Stage152-B 没有这个 86 维 contract。
- 复用 Stage152-B 时，先解析 `g1_dyn` 的 named input features 与 `token_flattened/proprioception` 的实际分块，再选择 decoder-input residual 或 hidden residual。
- 使用 X2-native lower-velocity 时，才讨论 86→122、89→125 的 observation 扩展。
- 未完成 named feature 分块审计前，禁止加载 Stage152-B 做新训练。
