# 第六轮结果卡：Future-Intent 腰髋协调 Adapter

日期：2026-07-28  
状态：完成；得到明确局部正信号，但未通过一致性门，保留 Stage208

## 一句话结论

提前 `0.6 s` 的上肢意图加 gait phase 确实能在部分 `0.30 m/s` 动作上大幅
降低航向和横漂，但 25 轮后低速总生存下降，并严重破坏一个留出 wave；因此它
不是可部署的通用 Adapter，也不应继续用同一目标盲目长训。

## 假设

Stage208 只能看到当前 arm `q/dq`，而上肢 nominal delay 约为 10.5 个控制帧。
若下层提前看到有界上肢意图和 gait phase，小型腰髋 residual 应比只看当前意图
或去掉 phase 更能预先协调重心。

## 干预

- Stage208-s2550 actor 全冻结；
- 当前上肢目标 14D、`t+0.6 s` 与当前目标差 14D、gait phase 4D；
- 32-unit MLP 输出 8 个固定协调模态；
- residual 只作用于左右髋 pitch/roll/yaw 与 waist yaw/roll；
- 硬上限 `0.10`，其余腰腿/上肢输出严格为 0；
- 上肢 contract 固定为 `scale=0.25`、`time_scale=0.5`、
  `excursion≤0.12 rad`、`velocity≤0.20 rad/s`；
- 三支从同一 Stage208 独立做 1/5/25 updates，不串联短训 checkpoint。

## 对照

1. `BASE`：冻结 Stage208；
2. `CURRENT`：只读当前有界上肢意图；
3. `FUTURE`：当前 + 未来差分 + phase；
4. `FUTURE-NOPHASE`：与 FUTURE 同容量，但 phase 置零。

正式物理面板为 6 motions × `vx=0.20/0.30 m/s`，包含 3 个训练分布动作、
真实 wave，以及 2 个留出动作；全部使用同一 nominal+delay、seed42、400
控制步。CPU 多环境面板只在自身坐标系内横向比较，不与旧单环境 trace 混算。

## 结果

### 等价门和结构门

- Stage208 actor/critic tensor 的加载差为 0；
- 零初始化 FUTURE 与旧 Stage5 单环境 Isaac trace 的 39 个数组最大差为 0；
- 三支 1/5/25 checkpoint 均无 NaN/Inf；
- 25-update 三支冻结 actor 的逐 tensor 差均为 0；
- 非允许动作 residual 严格为 0，允许动作 residual 未超过 `0.100001`。

### Gate5

| 分支 | 生存 | 总步数 | 平均航向峰值 rad | 平均横漂 m |
|---|---:|---:|---:|---:|
| BASE | 5/12 | 3176 | 0.6567 | 0.5259 |
| CURRENT | 5/12 | 3171 | 0.6428 | 0.5203 |
| FUTURE | 5/12 | 3170 | 0.6141 | 0.4735 |
| FUTURE-NOPHASE | 4/12 | 3115 | 0.6570 | 0.5114 |

FUTURE 没增加生存数，但总体方向指标优于两个容量匹配对照。低速真实 wave
在相差仅 2 帧的共同窗口内，航向改善 `0.112 rad`、横漂改善 `0.082 m`，
因此只解锁 25-update 验证，不宣称成功。

### Gate25

| 分支 | 生存 | 总步数 | 低速总步数 | 平均航向峰值 rad | 平均横漂 m |
|---|---:|---:|---:|---:|---:|
| BASE | 5/12 | 3176 | 962 | 0.6567 | 0.5259 |
| CURRENT | 5/12 | 3108 | 913 | 0.6421 | 0.5500 |
| FUTURE | 5/12 | 3218 | 945 | **0.5597** | **0.2822** |
| FUTURE-NOPHASE | 5/12 | 3125 | 970 | 0.7017 | 0.5706 |

FUTURE 的 aggregate 信号很强，且留出 `female_lift@0.30` 的生存由
214 增至 273 步；但是：

- 低速总生存 `962→945`，没有改善当前主要失败带；
- 留出 `wave_left@0.30` 的航向峰值 `0.510→1.068 rad`；
- 同一留出动作的倾角峰值 `0.299→0.509 rad`，越过 `0.45 rad` 安全门；
- 生存数仍为 5/12，没有增加一个完整 8 秒案例。

## 结论

1. “future intent 完全没有信息价值”被否定。它在正常速度的多个动作上形成
   了 CURRENT 和 NOPHASE 不能复制的方向改善。
2. “当前小型 Adapter 已形成通用安全接口”被否定。均值改善由动作子集主导，
   低速和留出 wave 的反例足以禁止晋级。
3. phase 有作用：去掉 phase 后总体指标比 BASE 更差；但有 phase 仍不能保证
   跨动作泛化。
4. 继续同一 reward 做更多 PPO 不具备足够价值。训练 reward/episode length
   的震荡不能替代固定面板，25 轮已展示明显能力交换。
5. Stage208 继续作为可保留基线；FUTURE-s2575 只保留为研究候选，不用于
   SONIC 注入、真机部署或“迁移已完成”的陈述。

## 下一步

若开启第七轮，不能只是把 25 延长到 200。需要新的可证伪机制，例如：

- 将腰髋修正按上肢意图的左右对称性/单侧性分模态，并限制跨动作外推；
- 把 heading/lateral 风险作为训练期显式约束，而非只在评估后发现；
- 增加真正随机的初态/执行器扰动与多 seed，验证收益不是单条确定轨迹；
- 保留 BASE fallback，并先在 shadow mode 判断 Adapter 何时应拒绝介入。

在此之前，本轮已到价值甜点位，停止继续扫轮数、hidden size、horizon 和
residual 上限。

## 诚实边界

- 上层仍是 oracle reference，不是实时 SONIC/衣服输出；
- 只在 Isaac CPU compatibility mode、单 seed 下验证；
- 多环境 CPU 与旧单环境存在数值差，因此只作同面板相对比较；
- NVIDIA 内核模块 `595.71.05` 与用户态 `595.84` 不一致，GPU 结论无效；
- 未连接真机、未调用 SDK、未发送任何执行命令；
- 旧 X2/SONIC 工程全程只读。

## 证据

- 预注册：`ROUND6_PREREGISTRATION.md`
- Gate5：`reports/stage6_future_intent/stage6_gate5_decision.{md,json}`
- Gate25：`reports/stage6_future_intent/stage6_gate25_decision.{md,json}`
- 结构审计：`reports/stage6_future_intent/*gate25_checkpoint_audit.json`
- 实现：`src/cwi_x2/future_intent*.py`、`hooks/sitecustomize.py`
- 固定面板：`scripts/run_stage6_panel_batch.sh`
- 单测：14/14 passed
