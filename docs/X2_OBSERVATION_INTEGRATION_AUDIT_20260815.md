# X2 六链目标的 observation 接入审计

日期：2026-08-15  
审计类型：只读接口审计，没有修改 SONIC/X2 旧 checkout，没有启动训练

## 已确认的当前接口

在官方 X2 `sole12` 任务配置中，静态探针观察到：

| 组 | 当前项 | 当前维度 |
|---|---|---:|
| policy | `base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_pos(31) + joint_vel(31) + actions(15)` | 86 |
| critic | policy 项 + `base_lin_vel(3)` | 89 |

配置入口为 `gear_sonic/envs/x2_velocity/flat_env_cfg.py` 的 `X2LowerObservationsCfg.PolicyCfg` 与 `CriticCfg`。两组都设置了 `concatenate_terms=True`。

六链契约的单帧目标误差是 6 个链节 ×（位置 3 + 旋转向量 3）= 36 维。因此直接把一帧目标追加到 observation 后：

- policy：`86 + 36 = 122`；
- critic：`89 + 36 = 125`；
- 若使用 H 帧历史：分别为 `86 + 36H` 与 `89 + 36H`。

第一阶段建议只做 H=1。在线衣服链路没有真正的未来信息，不能把离线未来帧直接伪装成部署输入；H>1 必须等因果预测器或固定延迟窗口定义完成后再做。

## 旧权重兼容结论

SONIC 的 `Actor` 包装器接收 `actor_obs`，再把 observation 交给可配置 backbone；`Critic` 单独接收 `critic_obs`。当前 `ppo_trainer.py` 已有 `_expand_appended_critic_checkpoint_features()`，能把旧 critic 的输入尾部零扩展并保持初始 value 不变，但没有对应的 actor 中性扩展函数。

因此，直接追加 36 维并加载旧 actor checkpoint 会触发第一层形状不匹配，不能静默截断、填零或忽略错误。

## 推荐的最小兼容方案

先不改变旧 actor 主干的 86 维输入：

1. 保留原 86→hidden 的 source trunk；
2. 新增 `future_goal_encoder(36H→hidden)` 分支；
3. 将该分支以零初始化的 residual 加到主干的第一个可训练 hidden 表示；
4. 当目标分支为零时，输出必须逐元素等于旧 checkpoint；
5. critic 可以先复用已有的中性 appended-feature 扩展，但必须记录 policy/critic 的输入契约分别为 86/89 或 122/125，不能混用。

如果现有 backbone 不暴露可插入的 hidden seam，再退回“从头训练新输入维度”的独立实验；这应作为新 checkpoint、新实验名和新对照，不得覆盖旧权重。

## 当前阻塞

实际 SONIC 源码 checkout 当前处于 detached HEAD `bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8`，且 `flat_env_cfg.py`、`ppo_trainer.py`、`actor_critic_modules.py` 等文件均有未提交修改，同时存在 `.sonic_runtime` 和未跟踪配置。因此本轮没有直接编辑该 checkout，也没有把它复制进 GitHub。

接入前必须先冻结一个可复现的源码 commit，或在 BFM-Zero 中创建明确的 overlay；否则即使训练结果变化，也无法判断是六链目标导致还是旧 checkout 的隐含改动导致。

## 下一道门

下一位 Agent 只能先完成以下工作：

- 固定 SONIC/X2 源码 commit 与依赖 manifest；
- 写一个旧 actor/critic checkpoint 的输入维度和 state-dict 形状审计；
- 在纯 PyTorch 中验证“目标分支全零时 adapter 等价于旧模型”；
- 再决定采用 residual adapter 还是全新输入维度训练。

在这四项完成前，不启动长训，不上传 checkpoint，不上真机。
