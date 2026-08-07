# Stage 8：X2 最小原生后端能力门禁

日期：2026-08-07  
状态：`contract_frozen / baseline_recheck_pending`  
范围：纯仿真；不接真机、不接真实 SONIC 上肢流、不启动长训

## 1. 阶段目标

冻结当前正式基线 Stage208，验证 X2 下层是否能在上层只提供**有界、慢速的人体
运动意图**时，稳定完成最小移动闭环：

```text
站立 → 起步 → 直行 → 左/右转 → 停止 → 重新站稳
```

这一阶段先验证“X2 原生后端能否兑现命令”，不把失败归因于衣服、PICO、SONIC
或新的迁移网络。

## 2. 冻结对象

### 正式基线

```text
checkpoint:
/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/
x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt

sha256:
df845bdbc78331fabdc9eba318fd244b81fa46f149631777b8abcd209866cac9
```

Stage208 是唯一正式 BASE。NEUTRAL05、FUTURE 和 FUTURE-NOPHASE 只能作为候选
或对照，不能因为某一个动作表现更好就替换正式基线。

### 三种方法

| 方法 | 含义 | gait phase 所属 |
|---|---|---|
| `BASE` | Stage208 原生下层，不加上层 Adapter | X2 后端内部 |
| `FUTURE_NOPHASE` | 有界 future upper intent，不提供 phase | X2 后端内部 |
| `FUTURE_PHASE` | 有界 future upper intent + X2 内部 gait phase | X2 后端内部 |

`HumanIntent` 只允许表达期望速度、航向、停止和有界上肢运动；gait phase、接触
状态、支撑脚、COM/DCM 等只能作为 X2 后端内部状态或 privileged 诊断，不能混入
设备无关接口。

## 3. 固定实验矩阵

机器可读版本见 [`configs/x2_native_backend_gate_v1.json`](configs/x2_native_backend_gate_v1.json)。

### 事件场景

| 场景 | 事件定义 |
|---|---|
| `stand` | 目标速度 0，保持站立 |
| `start_walk` | 0 → 0.20/0.30 m/s，观察起步前 2 s 和后续 4 s |
| `straight_walk` | 0.20/0.30 m/s 直行 |
| `turn_left` | 直行中加入左转航向意图，再恢复直行 |
| `turn_right` | 直行中加入右转航向意图，再恢复直行 |
| `stop` | 0.20/0.30 m/s → 0，观察减速和重新站稳 |

转向和停止必须是事件序列，不能用一条静态 reference 代替；否则测不到真正的
起步、接触切换和停止稳定性。

### 执行器域

第一版固定两个域，并记录其真实配置哈希：

1. `nominal_delay`：已有 X2 nominal 响应与延迟模型；
2. `stage208_randomized`：Stage208 使用的执行器随机化混合域（当前配置名含
   `ideal075_delay025`）。

若环境审计发现二者并非独立域，必须先改名并重新登记，不能用标签掩盖域重叠。

### 上肢扰动

| 模式 | 定义 | 作用 |
|---|---|---|
| `fixed` | 上肢保持默认或静态姿态 | 后端纯移动基线 |
| `slow` | 当前已验证工作点：scale `0.25`、time scale `0.5`、幅值 ≤ `0.12 rad`、速度 ≤ `0.20 rad/s` | 主要目标 |
| `fast` | 同幅值上限、time scale `1.0` | 压力测试，不预先承诺通过 |

### 重复与留出

正式统计至少使用 seed `17/42/73`，并配对初始姿态、横向速度和 yaw-rate。动作
事件相位也必须留出一组，不得只复用 Stage5/6 已见的连续固定速度窗口。

## 4. 统一指标与门禁

### 硬安全门

- 全事件无提前终止、无跌倒；
- 站立段支撑脚滑移不超过 `0.10 m/s`；
- 摆动脚按计划离地并在落地后重新建立接触；
- 停止后在 `2 s` 内回到稳定站姿；
- 没有未限制的 action delta 或 jerk 尖峰。

### 性能指标

- root 线速度 RMSE 与目标速度误差；
- 世界航向误差峰值和均值；
- 横向漂移峰值、时间均值；
- 接触时序、支撑脚滑移、摆动脚 clearance；
- action delta、action jerk；
- stop settle time；
- 上肢扰动前后性能下降。

所有指标都必须保留逐事件、逐 seed、逐执行器域结果；均值不能掩盖单个失败。

### 晋级规则

先判断 BASE 是否具备“可用后端”资格。若 BASE 自身在 nominal_delay 的起步、转向
或停止事件失败，Stage8 只能输出后端缺陷诊断，不能把 Adapter 的局部改善称为
迁移成功。

候选方法必须同时满足：

1. 生存数不低于 BASE，且不能让 BASE 已通过的留出事件跌倒；
2. 在所有执行器域和三个 seed 上无明显方向性反转；
3. slow 上肢扰动不突破安全门；
4. fast 扰动即使不能完整跟踪，也必须安全降级；
5. 至少在一个 held-out 事件上相对 BASE 有明确改善，且没有另一类事件的同量级回归。

## 5. Silver 数据规则

暂不大规模重定向 AMASS。只从这五类事件生成小规模动态 Silver：

```text
成功 X2 rollout / CEM-MPPI 轨迹
→ root、接触、foot slip、clearance、停止稳定性审计
→ 与测试事件和 seed 隔离
→ Silver dataset
→ 针对性训练
→ 完全独立门禁
```

成功 rollout 不能直接全部回灌训练；必须保留失败和 near-miss 标签，避免把当前
策略的错误闭环蒸馏回去。

## 6. 当前证据覆盖审计

截至本阶段开始，已有 Stage5/6/7 结果只覆盖“固定速度连续运动 + 上肢扰动”窗口，
没有完整的 start/turn/stop 事件，也没有统一的 foot slip、action jerk、stop settle
指标。因此不能把既有 `3/4` 或 `5/12` 结果解释成最小原生后端通过。

第一轮工作顺序固定为：

1. 完成执行器域和 command/event 语义审计；
2. 先跑 BASE 的事件矩阵；
3. 再跑 FUTURE_NOPHASE/FUTURE_PHASE 的同矩阵对照；
4. 只有 BASE 与候选的证据完整，才允许制作 Silver 或解锁短训。

## 7. 停止条件

- 若 BASE 在固定事件矩阵中无法完成基本起步/停止，转入后端 command contract 或
  native locomotion 修复，不扩大 LoRA；
- 若 BASE 可用但两个 Future 分支均无一致改善，停止该 Adapter 方向；
- 若成功只存在于训练动作而留出事件失败，停止继续加数据，优先做选择性介入；
- 未通过本门禁前，禁止真实 SONIC 上肢注入、真机部署和长训。
