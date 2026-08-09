# WBT Phase49：fresh-source exact-S7 最多 5-update pilot

## 裁决

**在 update 4 触发预注册硬停；update 5 未运行。**

共执行 `4 updates / 80 optimizer minibatch steps / 6144 transitions`。Gold 进入 optimizer 的样本数始终为 0，数值、KL、冻结参数与 survival/termination 门始终正常；真正触发停止的是动态 lunge 的跟踪方向。

## 假设

如果 Bronze-only faithful exact-S7 不只是一次更新的偶然数值可行，那么从原 SONIC source 出发，在最多五次更新内，每次更新后都应保持：

- KL `< 0.02`，loss/gradient 有限，冻结参数 hash 不变；
- lunge survival 下降不超过 `0.1 s`、termination 增量不超过 `0.05`；
- lunge 至少一项跟踪误差改善，且没有误差恶化超过 `10%`；
- native Gold survival/tracking 不超过预注册退化门。

任一失败立即停止，不能用整体 PPO reward 上升覆盖固定动作门。

## 干预

- seed `0`；每次 `64×24=1536 transitions`；每次 `5 epochs×4 minibatches=20 optimizer steps`；
- exact-S7 actor `0/2/4/6/8/10/12`、critic `2/4/6/8/10`；
- optimizer 只见三条 kinematic Bronze；
- 每次进程恢复上一段 optimizer、scheduler、trainer 和 environment checkpoint state，并使用五次更新的统一 scheduler horizon；
- 每次只执行一个 outer update，随后独立跑 lunge 与 native Gold deterministic 零优化门。

## 结果

| update | KL | train mean length | lunge survival | lunge 最佳误差变化 | Gold 最坏误差变化 | 裁决 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.01670 | 11.73 | 0.22 s | -0.184% | +1.19% | 通过 |
| 2 | 0.01588 | 15.33 | 0.22 s | -0.259% | +2.36% | 通过，最佳 |
| 3 | 0.01708 | 16.39 | 0.22 s | -0.091% | +4.27% | 通过但收益收窄 |
| 4 | 0.01575 | 19.15 | 0.22 s | **+0.116%** | +5.56% | **硬停** |

update 4 中六项 lunge tracking error 全部变差，最小恶化也为 `0.116%`；因此违反“至少一项改善”的方向门。它的 KL、finite、冻结 hash、lunge survival/termination 以及 Gold 门都仍通过，但这些不能覆盖该硬失败。

这里得到一个重要而诚实的结论：**PPO 训练平均 episode length 从 11.73 增至 19.15，并不等于目标动态 lunge 在改善。** 固定动作诊断成功识别出 aggregate 指标掩盖的偏离，说明这次硬门有实际价值。

## 权重裁决

- source：`update01/source_B0.pt`；
- 最佳通过权重：`update02/last.pt`，因为它在所有过门模型中 lunge 改善最大，Gold 仍过门；
- 最后拒绝权重：`update04/last.pt`，只保留作硬停证据，禁止续训；
- update 1、3 的 `last.pt` 属冗余中间权重，可清理；所有 JSON、配置与日志保留。

“最佳”仅指本轮 **optimizer-sanity** 多目标门中的最佳，不代表机器人已会 lunge、行走或可部署。所有更新的 lunge survival 都只有 `0.22 s`。

## 结论与下一步

faithful exact-S7 已证明可以连续积累三次受控更新，但第四次开始偏离动态目标；因此不能进入 update 5，更不能进入长训。

不要继续这条 checkpoint chain。后续应利用这次结果重新设计训练数据或动态诊断，使 lunge survival/方向真正提升后再讨论更长训练，而不是依据平均 reward/episode length 继续刷 update。
