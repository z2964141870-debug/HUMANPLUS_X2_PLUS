# 第二轮裁决：Actor 目标梯度是否真的冲突

日期：2026-07-28  
范围：纯仿真、无真机、旧工程只读  
目标：决定是否证据解锁 PCGrad / GCR-PPO

## 一句话结论

在当前 Reward Contract、Stage152-B LoRA 注入范围和 4 条步态训练分布内，locomotion 与 imitation/upper 的 actor 梯度高度同向；PCGrad 缺少成立前提，不应继续测试。

## 假设

第一轮已证明语义双 critic 让 value estimation 更准确，却没有形成物理 Pareto，并明显破坏上肢保持。一种可能解释是：两个 head 的 advantage 最终仍进入同一 actor，产生持续负梯度冲突。

若该假设成立，应观察到：

- 多 seed 下频繁出现负 gradient cosine；
- I25 比初始点冲突更严重；
- 25 轮训练轨迹中负冲突持续存在，而非孤立噪声；
- 语义分组的冲突结构与 shuffled control 不同。

## 干预

在 ZHY import overlay 中加入逐 reward-head actor-gradient probe：

- 直接从同一个 PPO graph 恢复各 head 的 clipped policy loss；
- shared entropy 不归入任何 head，避免人为制造共同梯度；
- 对 14 个 trainable actor LoRA tensor、共 123,432 个参数分别求梯度；
- 同时记录 microbatch cosine、整轮 aggregate cosine、逐层 cosine 和范数；
- policy-head loss 必须重构原 policy loss，误差门为 `1e-6`。

探针没有修改旧 trainer。边界矩阵使用 optimizer 创建后显式归零，并逐位检查 actor 参数不变。

## 对照

边界矩阵共 12 个独立 probe：

```text
E1 semantic × {INIT, I25} × seeds {0,1,2}
N1 shuffled × {INIT, I25} × seeds {0,1,2}
```

每个 probe：32 env × 24 steps × 4 PPO minibatches；ideal/filter 各 50%。  
另完整复跑 E1 seed0 25 iterations × 12 microbatches，得到 300 个训练过程观测点。

## 工具正确性

- 新增/原有单测合计 `9/9 passed`；
- 12/12 边界 probe 的 actor parameter max abs delta 为严格 `0.0`；
- 12/12 均未生成 checkpoint；
- 边界 probe 最大 policy-loss reconstruction error 为 `5.96e-08`；
- 25 轮轨迹最大 reconstruction error 为 `8.94e-08`；
- 带探针复跑与原 E1 的 reward、episode length、KL、actor/critic grad、replay RMSE 和 termination 核心字段逐轮完全一致。

## 三 Seed 边界矩阵

| 分组 / checkpoint | aggregate cosine mean | aggregate min | micro mean | micro min | negative microbatches |
|---|---:|---:|---:|---:|---:|
| E1 semantic / INIT | 0.9417 | 0.8874 | 0.9393 | 0.8500 | 0/12 |
| E1 semantic / I25 | 0.8958 | 0.8916 | 0.8944 | 0.7710 | 0/12 |
| N1 shuffled / INIT | 0.9649 | 0.9258 | 0.9665 | 0.8796 | 0/12 |
| N1 shuffled / I25 | 0.9711 | 0.9458 | 0.9696 | 0.9359 | 0/12 |

E1 到 I25 后对齐程度有所下降，但仍然是强正对齐，不是冲突。随机分组反而更接近共线，这说明 shuffled heads 更像同一信号的任意拆分，也解释了它为什么不能作为有意义的目标隔离。

逐 LoRA 层审计没有发现被全局均值掩盖的负冲突。E1-I25 三 seed 的最低 active-layer cosine 分别为 `0.859 / 0.823 / 0.757`，仍全部为正。

## 完整 25 轮梯度轨迹

| 指标 | 结果 |
|---|---:|
| iterations | 25 |
| microbatches | 300 |
| aggregate cosine mean | 0.8972 |
| aggregate cosine min | 0.6466 |
| aggregate cosine max | 0.9799 |
| microbatch cosine mean | 0.8534 |
| microbatch cosine min | -0.4292 |
| negative microbatches | 2/300 = 0.67% |
| iterations with any negative microbatch | 2/25 |
| iterations with negative aggregate gradient | 0/25 |

仅 iteration 22、23（从 1 计数）的各一个 microbatch 为负，其余 298 个均为正。PCGrad 只在 dot product 为负时投影，因此在当前轨迹中最多影响 `0.67%` 的 microbatch，且不会改变任何整轮 aggregate direction。

## 结果

当前“E1 上肢退化来自共享 actor 的持续负梯度冲突”假设被推翻。更符合证据的解释是：

1. 当前 `imitation_upper_regularization` 不是纯 upper objective，包含 whole-body compound terms；
2. 训练只使用 4 条步态，upper head 缺少独立、多样的上肢任务分布；
3. 两个 head 的 normalized advantage 实际产生高度相似的 actor 更新方向；
4. critic 更准只能降低 value fitting noise，无法把不充分的目标语义变成能力保持约束；
5. held-out wrist error 与训练 head reward 并非同一个严格约束，因而训练统计变好时仍可退化。

## 结论

**不解锁 PCGrad / GCR-PPO 5/25 对照。**

继续做梯度投影的预期收益很低，并且会把“偶发 2/300 负 microbatch”夸大为主要矛盾。第一轮 E1 的失败不是 critic 容量不足，也不是当前两 head 的持续 actor 梯度互斥。

这同时把下一条主线收窄为：

```text
先做 CWI 的数据使用解耦 / 纯 objective contract
        或
回到 reference feasibility 与动态可执行性
```

若继续 CWI，相邻的最小实验应是数据流因果对照，而非换优化器：locomotion head 只看干净动态步态，upper head 额外看站立挥手/伸手等上肢数据，并用 held-out wrist preservation 作硬门。

## 证据入口

- 12 个边界 probe：`results/objective_gradient_{e1,n1}_{init,i25}_seed{0,1,2}_v2.jsonl`
- 25 轮轨迹：`results/e1_i25_gradient_trajectory_seed0_v1.jsonl`
- 实现：`src/dcpeft_gradient_geometry.py`、`src/sitecustomize.py`
- 启动脚本：`scripts/run_objective_gradient_probe.sh`
