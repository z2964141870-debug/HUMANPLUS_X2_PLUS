# Phase 3-I25 最终裁决：CWI 式双 Critic 的价值边界

日期：2026-07-28  
起点：Stage152-B `model_step_000200.pt`  
预算：B1/E1/N1 各 25 iterations，seed 0，随后运行同一 4-motion × 4-domain 门禁

## 一句话结论

语义双 critic（E1）显著改善了 value 分解和训练期统计，但没有改善最终物理 Pareto，且四域上肢 wrist error 相比 B0 恶化约 26%；因此按预注册规则停止扩大训练，不解锁 200×3 seeds。

## 假设

Stage152-B 的单 critic 同时估计 locomotion、upper-body imitation 与 whole-body regularization，可能造成 value/advantage 估计干扰。若语义分组有效，E1 应同时满足：

1. value quality 优于单 critic 与随机分组；
2. 物理门禁优于 B1；
3. 不以牺牲 upper-body preservation 换 locomotion；
4. 优于错误分组 N1。

## 干预

- B1：原 scalar reward + single critic；
- E1：等价 vector reward + semantic dual critic；
- N1：等价 vector reward + 固定 seed shuffled dual critic；
- E1/N1 均由 single critic 做 aggregate-preserving equal split 初始化；
- actor、训练数据、物理域、PPO 超参数、LoRA 范围与预算保持一致；
- 所有分支均从 Stage152-B 独立启动，未串联短训 checkpoint。

## 等价性与数值健康

- E1 reward reconstruction 最大误差：`5.96e-08`；
- N1 reward reconstruction 最大误差：`1.19e-07`；
- 两者均低于 `1e-6` 硬门；
- KL、actor/critic gradients、fixed-replay action drift 均有限；
- 未发现 reward broadcast、head shape 错位、重复 LoRA 合并或有效 actor 学习率放大。

## 25 轮训练统计

下表为末 5 iterations 均值；critic 指标为最终 iteration。

| 指标 | B1 single | E1 semantic dual | N1 shuffled dual |
|---|---:|---:|---:|
| episode reward | 4.950 | **5.381** | 4.343 |
| episode length | 71.55 | **76.42** | 62.40 |
| KL | 0.001511 | 0.001681 | 0.001787 |
| fixed replay action RMSE | **0.003438** | 0.003647 | 0.003613 |
| actor grad norm | 9.373 | **8.398** | 8.714 |
| critic grad norm | 5.828 | 3.755 | **1.870** |
| training anchor error | 0.3096 | 0.2403 | **0.1645** |
| timeout fraction | 0.771 | 0.814 | **0.882** |
| foot termination fraction | 0.138 | 0.182 | **0.086** |
| value explained variance | 0.878 | **0.832 / 0.830** | 0.734 / 0.647 |
| TD residual std | 0.181 | **0.080 / 0.160** | 0.225 / 0.196 |
| pre-update value loss | 0.183 | **0.030 / 0.106** | 0.165 / 0.148 |

机制层面，E1 相比 N1 的两个 value head 具有更高且更一致的 explained variance、更低 TD residual 和更低 value loss。这支持“语义 reward decomposition 能让 critic 更容易估计”的局部机制假设。

## 四域物理门禁

汇总口径为 4 motions × ideal/filter/delay/noise，共 16 条。

| 模型 | strict pass | stable | progress | contact | foot | root XY RMSE (m) | progress ratio abs error |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 frozen | 1/16 | **16/16** | 2/16 | **12/16** | **9/16** | 0.2370 | 2.8263 |
| B1-I25 | 0/16 | 15/16 | 1/16 | **12/16** | 8/16 | 0.2191 | 2.1499 |
| E1-I25 | **1/16** | 12/16 | **3/16** | 11/16 | 7/16 | 0.2050 | 2.2012 |
| N1-I25 | 0/16 | **16/16** | 1/16 | **12/16** | 7/16 | **0.1954** | **1.9288** |

E1 的 strict pass 只恢复到 B0 的 1/16，且从 B0 的 filter 域转移到 ideal 域；它没有扩大总通过数。E1 的 stable、contact、foot 均低于 B0，连续 root/progress 指标也没有优于 N1。因此不能声称语义分组带来总体物理改善。

## 上肢能力保持

由四域 rollout trace 的 4,160 个样本统一重算：

| 模型 | upper mean (mm) | wrist mean (mm) | wrist p95 (mm) | anchor mean (mm) |
|---|---:|---:|---:|---:|
| B0 frozen | 48.50 | 56.23 | 132.09 | 173.13 |
| B1-I25 | 50.09 | 55.53 | 133.46 | 156.97 |
| E1-I25 | **70.31** | **71.07** | **166.28** | 150.91 |
| N1-I25 | 47.97 | 54.38 | 131.80 | **146.34** |

E1 wrist mean 相比 B0 增加 `26.38%`，明显超过任务卡允许的 `5%` degradation。它因此触发“以牺牲另一目标获得改善”的停止条件。B1/N1 的 upper preservation 均明显好于 E1。

## 结果

1. 工程成功：vector reward、single compatibility、dual checkpoint 保存/恢复、逐头 GAE 与诊断全部成立。
2. 机制有限成立：E1 critic 的 value quality 明显优于 shuffled N1。
3. 方法当前不成立：更干净的 critic 没有转化为 actor 的物理 Pareto 改善。
4. 语义分组不是无效噪声，但当前 group 仍含不可拆的 whole-body compound terms，无法成为真正纯净的 locomotion/upper 评分员。
5. 最终仍是同一个 actor 接收合成 advantage；multi-critic 只减少 value estimation interference，并不自动消除 actor gradient conflict。

## 结论

不解锁 Phase 4 的 200 iterations × 3 seeds。原因不是“25 轮还不够长”，而是 E1 已出现预注册的能力交换反例：训练曲线更好、critic 更准，但 upper-body preservation 和稳定性更差。盲目拉长预算会把机制指标误当任务进展。

这个负结果不否定完整 CWI。当前实验只检验了 critic decomposition；CWI 中的数据解耦、style discriminator 和更明确的 objective isolation 尚未被复现。

## 下一步

优先级从高到低：

1. 在不训练的情况下测量各 objective 对共享 actor/LoRA 参数的梯度余弦，确认是否存在持续负梯度冲突；
2. 只有测得稳定负冲突，才做 PCGrad/GCR-PPO 类 actor 梯度融合的最小 5/25 对照；
3. 若梯度并不冲突，则停止 CWI 主线，回到 reference feasibility / data-use decoupling；
4. 若以后重做 dual critic，应先拆掉 compound whole-body term，构造真正独立的 locomotion、upper 和 style/regularization contract，而不是继续调 critic 权重。

## Checkpoint 证据

- B1-I25 SHA256：`8b1850b4e1ac72a5ef22017afcd180958f7764abc8d09c94dad573bef413d6d5`
- E1-I25 SHA256：`ee292c7a6b5892d01520e9d477800a099166b1fac33a00233c9ce5dfd1553dfc`
- N1-I25 SHA256：`9504396a2bd1063d6e1950c0497881c80d5420cd88d46a59b27ebb98035be23e`

这三个 step-25 checkpoint 属于预注册的 final evidence，不是冗余中间点，暂时保留。
