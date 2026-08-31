# 衣服六链目标的现有 SONIC adapter 接入点

日期：2026-08-15  
结论类型：代码考察与接口设计；尚未修改 dirty SONIC checkout

## 已发现的可复用机制

实际 SONIC checkout 已经包含以下未提交实现：

- `gear_sonic/trl/modules/universal_token_modules.py`：`DirectContactActionAdapter`；
- 同文件：`TemporalBilinearActionAdapter`；
- `gear_sonic/trl/utils/x2_checkpoint_semantic_mapping.py`：支持 `g1_dyn` 输入的语义分块和命名特征后缀扩展；
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_phuma_direct_contact_adapter.yaml`：在 `g1_dyn` 解码器边界注入零初始化、有界动作残差的已有实验配置。

这条机制已经具备：

1. 只新增 adapter 分支，不破坏冻结主干；
2. 最后一层零初始化，初始输出等于旧 decoder；
3. 只对选择的输出关节产生残差；
4. 残差幅度有硬上限；
5. 可对输入特征做命名和连续 span 校验；
6. 支持普通线性或 state×temporal interaction 两种分支。

## 建议的衣服接入形态

把六链误差编码器的输出登记为一个新的 tokenizer feature：

```text
x2_extremity_goal : 36*H
```

并将 `g1_dyn` 输入从：

```text
[token_flattened, proprioception]
```

扩展为：

```text
[token_flattened, proprioception, x2_extremity_goal]
```

第一阶段使用 H=1（36 维）；未来窗口不能用离线未来帧冒充实时衣服输入，必须先定义因果预测或固定延迟语义。

对于 X2 full action order，现有配置使用的 15 个下肢/腰部输出选择为：

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19, 20]
```

该索引必须在实际 live action names 上再次校验，不能从 G1 的关节顺序复制。

## 为什么这条路线比直接扩展 actor observation 更合适

- Stage152-B 的 `g1_dyn` 输入是 1062 维，critic 是 1745 维，不是 X2-native lower policy 的 86/89 维；
- `x2_checkpoint_semantic_mapping` 已经处理过“保留旧 decoder 前缀、把新命名特征放在末尾”的形状映射；
- adapter 可以只改变 X2 下肢 action residual，保留上肢和 token 主干的源能力；
- 目标输入是当前机器人位姿与衣服目标的相对误差，已经包含状态信息，第一版不必再把隐式 simulator state 暴露给 adapter。

## 必须先解决的实现问题

1. 在 `gear_sonic/envs/manager_env/mdp/observations.py` 中定义 `x2_extremity_goal` 的来源和形状；
2. 明确衣服目标来自离线回放、同步后的在线命令，还是一个可学习预测器；
3. 在 tokenizer manager 中登记 36 维 feature，并确认它与 `g1_dyn` 输入 span 连续；
4. 用 `x2_checkpoint_semantic_mapping` 生成旧→新 decoder 的零填充映射；
5. 用纯 PyTorch 检查 zero adapter 下旧 checkpoint 输出逐元素一致；
6. 只在上述检查通过后做短时仿真，不启动长训。

当前 dirty checkout 中这些文件均有未提交改动，因此接入必须先冻结源码 commit 或建立 overlay。不能直接在该 checkout 上继续叠加修改。
