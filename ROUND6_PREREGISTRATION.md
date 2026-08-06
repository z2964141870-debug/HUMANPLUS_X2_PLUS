# 第六轮预注册：Future-Intent Lower Coordination Adapter

日期：2026-07-28  
状态：待实现；先等价门，后 1/5/25 级联

## 最终命题

Stage5 已证明：Stage208 可以反应式读取当前 arm `q/dq`，但在上肢执行器约
10.5 个控制帧延迟和窄吸引域下，固定滤波与事后回退都不能形成通用安全接口。

本轮只检验：

> 给冻结下层一个可部署的、提前 0.6 秒的有界上肢意图，是否比只给当前意图
> 或不给 gait phase 更能减少跌倒、航向和横漂。

用户已允许最终动作有一定延迟，因此部署时可将人体输入缓存 0.6 秒；本轮不把
真正未来不可知信息伪装成实时可用信号。

## 架构

```text
Stage208 93-D observation ───────────────→ frozen base actor

current bounded upper target (14)
+ target(t+0.6s)-target(t) (14)
+ existing gait phase/contact clock (4)
                    ↓
              32 hidden MLP
                    ↓
          8 fixed waist/hip modes
                    ↓
       bounded normalized residual ≤ 0.10
                    ↓
          base action + coordination
```

- 腿模板、上肢路径、Stage208 actor 主体全部冻结；
- Adapter 只影响左右 hip pitch/roll/yaw 与 waist yaw/roll；
- 不直接修改膝、踝或上肢；
- 最后一层权重和 bias 均为零初始化；
- upper intent 全零时 residual 由硬 gate 保证严格为零；
- critic 保持原 93-D contract，不读取 future intent。

## 容量匹配对照

三支 Adapter 的网络、参数量、输出 basis、预算完全相同：

1. `CURRENT`：只使用当前有界 upper target，未来差分强制为零；
2. `FUTURE`：使用当前 target + 0.6 秒 future delta + gait phase；
3. `FUTURE-NOPHASE`：future 信息相同，但 Adapter 的 gait phase 强制为零。

另保留冻结 Stage208 `BASE`，不训练，仅作绝对回归基线。

只有 `FUTURE` 同时优于 CURRENT 与 FUTURE-NOPHASE，才能把收益归因于
“未来意图与步态相位协调”，而不是多了参数或继续 PPO。

## Upper 训练集

训练只使用已有 reference 的 14 个肩/肘/腕通道：

1. ACCAD Male1 swing arms；
2. ACCAD Female1 swing；
3. ACCAD Male2 box lift；
4. BMLrub rub073 knocking；
5. KIT wave both09；
6. KIT shower right arm03。

统一控制契约：

- scale `0.25`；
- time scale `0.5`；
- excursion `0.12 rad`；
- target velocity `0.20 rad/s`；
- nominal upper actuator response 保持；
- clip 随 env/reset 分配，循环时仍经 slew-rate 限制。

## 留出门禁

不参与训练的主要留出动作：

1. 实机采集真实 wave（start 5 s）；
2. KIT wave left01；
3. ACCAD Female1 lift box。

同时保留 Stage5 的 swing/knocking/box 三条回归动作。正式面板覆盖
`vx=0.20/0.30 m/s`、nominal+delay、每条 8 秒。

## 训练合同

- 起点：Stage208-s2550；
- actor base 冻结，只有 coordination Adapter、critic、std 可训练；
- 速度训练范围：`0.20–0.45 m/s`；
- heading command、gait template、reward、collision、PD 与 Stage208 匹配；
- actuator mixture：75% ideal + 25% nominal+delay；
- 首轮：32 env，seed42，1/5/25 iterations；
- 不在本轮扫描 hidden size、horizon、residual scale 或 reward 权重。

## 级联门

### Gate 0：零初始化等价

- adapter-free Stage208 checkpoint 可完整载入；
- base actor tensor 全部匹配；
- 同一 upper rollout 下，BASE 与零初始化 FUTURE 的 mean action 与所有物理
  trace 最大差 `≤1e-6`；
- intent observation 维数、顺序、单位写入 manifest。

不通过则禁止训练。

### Gate 1：1 iteration

- 无 NaN/Inf；
- base actor 参数漂移严格为 0；
- Adapter residual 不超过 0.10；
- checkpoint 可重载；
- CURRENT/FUTURE/FUTURE-NOPHASE 三支均从同一起点独立运行。

### Gate 5

运行固定的 6-motion×2-speed 面板。只有 FUTURE：

- 生存数不低于 BASE；
- 在 Stage5 failed wave 或低速条件至少改善一个；
- heading/lateral 的总体改善同时优于 CURRENT 和 FUTURE-NOPHASE；
- 不破坏原 3 条正常速度通过动作；

才解锁 25 iterations。

### Gate 25

要求 FUTURE 相对 BASE、CURRENT、FUTURE-NOPHASE 形成一致的安全/方向优势；
训练 reward 或 episode length 单独上升不构成晋级。

## 停止与诚实边界

- 若三支 Adapter 都相似改善：只能说明 disturbance training 有效，不能声称
  future intent 有效；
- 若 CURRENT 最好：未来预告没有信息增益，转纯鲁棒反馈；
- 若 FUTURE-NOPHASE 最好：phase conditioning 可能有害；
- 若全部劣化：保留 Stage208，不继续扩大网络或训练轮数；
- 本轮 oracle intent 仅验证接口，仍不等于完成 SONIC→X2；
- 当前 Isaac compatibility mode 只支持方法学验证，不作吞吐量结论。
