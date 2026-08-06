# 第三轮任务卡：数据用途解耦与腰腿动作所有权

日期：2026-07-28  
状态：**完成；得到有限正结果，但禁止进入长训**

## 本轮目标

回答两个容易混在一起的问题：

1. 第一轮中观察到的上肢误差，究竟是策略忘了上肢，还是腿部闭环失稳后的连带现象？
2. 如果只允许新 LoRA 改腰部和下肢，能否保护 Stage152-B 的上肢能力，并让四域步态随训练持续改善？

## 数据契约

| 数据组 | 用途 | 是否允许训练 | 说明 |
|---|---|---:|---|
| 4 条官方 X2 原始/镜像步态 | 训练 + 四域评估 | 是 | 当前唯一进入本轮 PPO 的动态数据；世界 root 仅作有缺陷的参考，不视作动力学真值 |
| 3 条 fixed-feet 上肢动作 | 独立能力评估 | 否 | 锁定 root、腿、腰，隔离上肢语义；不拿来改变训练分布 |
| 12 条 gait × upper 合成动作 | 压力测试 | 否 | 只验证组合能力，属于 synthetic probe，不能伪装成真实训练数据 |

完整机器可读契约见 [round3_data_and_projection_manifest.json](manifests/round3_data_and_projection_manifest.json)。

## 实验 A：上肢退化是否是独立遗忘

假设：
如果 E1 在 fixed-feet 动作上仍明显变差，说明上肢能力本身被遗忘；如果只在步态中变差，则更可能是下肢/根部失稳的闭环连带效应。

干预：
不训练。对 B0 与 E1-I25 运行相同 fixed-feet 上肢动作，先做 ideal 全程回放，再做 filter 5.2 秒回放。

对照：
同一动作、同一初态、同一评估器；唯一差异是 checkpoint。

结果：
filter 域两者均稳定完成 `3/3`。E1 相比 B0：wrist mean `+1.78%`、wrist p95 `-0.79%`、upper mean `+1.64%`、anchor p95 `-5.94%`，均在 5% 能力保持门内。

结论：
第一轮 gait filter 中约 `+97.6%` 的腕误差不能解释为上肢独立遗忘，主因是 locomotion/contact 失败后的任务空间连带偏差。上肢数据面板有价值，但给训练额外塞上肢 rehearsal 并没有被证据支持。

下一步：
把 fixed-feet 上肢集保留为独立回归门，不加入 PPO 训练。

## 实验 B：构造真正的腰腿限定适配

假设：
只在 decoder 最后一层做 output mask 并不能保护上肢，因为前层 LoRA 仍会经共享网络泄漏；必须限制整个 decoder 相对冻结基座的最终 action delta。

干预：

1. 将 Stage152-B 的 7 层 actor LoRA 与 7 层 critic LoRA 物化进普通线性层，得到共同基座；
2. 从该基座重新零初始化 LoRA；
3. 在 decoder 边界计算“适配输出 − 冻结输出”，再按动作通道投影；
4. LBP 只开放腰部 3 DOF + 双腿 12 DOF，共 15 DOF。

对照：
FBP 使用完全相同的物化基座、数据、critic、学习率和 decoder 投影，只是开放全部 29 个非头部 DOF。

结果：

- actor/critic 物化各 7 层，旧 checkpoint 未修改；
- FBP/LBP 起点首步 `4 env × 31 action` 的 mean action 最大差和平均差均为严格 `0`；
- 11 个物化/掩码/脚本测试全部通过；
- LBP 的 runtime contract 明确选择 15/31 action，FBP 选择 29/31 action。

结论：
本轮建立了一个可复用的“动作所有权”机制。它限制的是整个适配 decoder 的最终残差，而不是只掩盖最后一层参数。

下一步：
用 5-step 和 25-step 匹配训练验证它是否带来物理收益。

## 实验 C：匹配的 5-step 四域预筛

假设：
若共享上肢动作自由度是早期不稳定的重要来源，LBP 应在 nominal/filter/delay 域同时改善 root、接触和足端，并保持上肢。

干预：
LBP 仅允许 15 个腰腿动作通道接收 LoRA delta。

对照：
FBP 允许 29 个非头部动作通道；其余完全一致，均从物化基座独立训练 5 steps。

结果：

| 域 | FBP：strict / stable | LBP：strict / stable | root RMSE FBP→LBP | progress-ratio error FBP→LBP | 接触 FBP→LBP |
|---|---:|---:|---:|---:|---:|
| ideal | 1/4 / 4/4 | 0/4 / 4/4 | 0.112→0.121 m | 0.811→0.641 | 2/4→0/4 |
| filter | 1/4 / 4/4 | 1/4 / 4/4 | 0.226→0.190 m | 2.293→2.207 | 4/4→4/4 |
| delay | 0/4 / 4/4 | 0/4 / 4/4 | **0.240→0.176 m** | **2.870→1.206** | **2/4→4/4** |
| noise | 0/4 / 4/4 | 0/4 / 4/4 | 0.217→0.252 m | 2.780→2.388 | 3/4→2/4 |

四域 gait 上肢对比均满足 5% preservation 门。训练 KL 也由 FBP 均值约 `7.58e-4` 降到 LBP 的 `4.62e-4`。

结论：
LBP 没有提高总 strict 数，但在最关键的 `nominal+delay` 域同时改善 root、进度误差、接触与足端，且未直接牺牲上肢。它满足“扩到 25-step”的预注册连续信号门。

下一步：
FBP/LBP 同时扩到 25-step；禁止只延长看起来更好的 LBP。

## 实验 D：25-step 是否保持向上趋势

假设：
如果动作所有权是主修复，LBP 的 5-step 收益应在 25-step 继续或至少不反转。

干预：
FBP/LBP 各自从同一物化基座独立训练 25 steps，随后运行完全相同的四域固定起点回放。

对照：
同实验 C；仅训练预算从 5 增至 25。

结果：

| 分支 | train reward | train episode length | 四域 strict | 四域 stable | delay root RMSE |
|---|---:|---:|---:|---:|---:|
| FBP-25 | 0.551→4.289 | 8.0→61.24 | 1/16 | 15/16 | 0.267 m |
| LBP-25 | 0.551→4.546 | 8.0→63.63 | 0/16 | 14/16 | **0.402 m** |

LBP-25 的 delay 域从 5-step 的稳定 `4/4` 退到 `2/4`，两条动作末段跌倒；wrist mean 相比 FBP 在 delay 域恶化 `36.2%`。noise 域则出现相反方向的改善，说明收益不是跨域一致的。

更关键的是：两支 `reward/episode_length` 从头到尾基本横盘在约 `0.063–0.071`。训练 reward 的大部分增长来自 episode 活得更久，而不是单位时间跟踪变准；训练中的世界 anchor 误差也没有随轮数单调下降。

结论：
**动作所有权是有效的结构安全工具，但不是当前长训单调改善的主解。** 它能在 5-step 短窗改善 nominal+delay，并在相同 decoder 输入下阻止 LoRA 直接改手臂；但下肢一旦把闭环带入不同状态，任务空间手部仍可因全身姿态偏移而变差。25-step 结果正式否定“仅靠腰腿投影即可解锁长训”。

下一步：
不运行 200/1000-step。先修训练契约和模型选择逻辑。

## 本轮总体裁决

### 已证实

- 上肢独立能力并没有像原 gait 指标暗示的那样被严重遗忘；此前的大幅腕误差主要是下肢闭环失败的连带结果。
- decoder 边界 action-delta projection 能严格定义新适配器的动作所有权。
- 腰腿限定在 5-step 的 nominal+delay 连续指标有真实正信号。
- 当前训练 reward 上升主要表示 survival 增长，不能代表世界步态越来越准确。

### 已否定

- 继续增加 critic、使用 PCGrad、额外上肢 rehearsal，均没有当前证据支持。
- 仅把 LoRA 限制到腰腿不能保证 25-step 后四域持续改善。
- 不能依据训练 reward/episode length 解锁长训。

### 下一阶段的硬要求

1. 保留 LBP 投影作为安全 contract，而不是性能方法；
2. 训练选择指标必须加入固定起点的世界进度、接触周期、足端和跌倒门，且与训练 reward 分开记录；
3. 不直接把不可信的高频 raw root 当更强硬的逐帧真值；应优先设计带宽受限的路径走廊、事件/接触相位或 locomotion task command；
4. 每 5–10 steps 做 deterministic panel 并早停，先证明 held-out 四域趋势与训练趋势同向，再谈 200+ steps；
5. 任何新方案必须同时报告 ideal/filter/delay/noise，禁止只挑一个域。

## Checkpoint 处置

- 保留共同基座：`checkpoints/stage152_B_materialized_v1.pt`；
- 暂留信息量最高的 LBP-5；
- FBP-5、FBP-25、LBP-25 均可由共同基座低成本重建；报告与 SHA 落盘后已删除，共释放 `463,406,917 B`；
- 25-step 两支均不得作为部署或续训基座。
