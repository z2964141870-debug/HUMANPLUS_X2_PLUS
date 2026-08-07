# SONIC/G1 → Agibot X2 迁移阶段总览

日期：2026-07-13

> 编号说明：项目早期文档中的 Stage 0–3 是临时路线编号；本文把阶段 0–9 按实际工程顺序统一重排，阶段 10–13 保留已有正式实验含义，并据此定义阶段 14–15。

## 第 0 阶段：先把目标定清

目标不是从零训练一个 X2 控制器，而是把已有的 SONIC/G1 全身动作跟踪能力迁移到 X2：

```text
人体动作 / 参考动作
        ↓
SONIC 原有动作理解与控制先验
        ↓
适配 X2 的身体结构、关节与执行器响应
        ↓
X2 在仿真中稳定跟踪动作
        ↓
后续再接入 11 IMU 智能服装遥操作链路
```

先确定了三个边界：

- X2 使用 31DOF；G1 原策略是 29DOF body 版本，不涉及灵巧手。
- X2 多出的两个头部关节不参与迁移动作控制，锁定默认位，也不进入迁移指标。
- 当前只做离线数据与 IsaacLab/MuJoCo 仿真，不连接或控制真机。

阶段结论：**完成。** “头”和“灵巧手”从一开始就不属于当前主要问题。

## 第 1 阶段：搭好 G1 到 X2 的身体接口

要先解决两个机器人关节数量、顺序、名称和身体比例不同的问题：

```text
G1 action[29]
        ↓  按 joint name 对齐，而不是按数组下标硬拷贝
X2 body action[29] + locked head[2]
        ↓
X2 action[31]
```

完成内容：

- 建立 X2 31DOF joint order、official qpos62、MuJoCo/IsaacLab DOF 映射。
- 修复 actuator、body 和 DOF selector 串位。
- 修复 checkpoint 形状不同时按 tensor 前缀复制的错误，改成严格语义映射。
- 任何无法识别的映射都 fail closed，不再悄悄错接。

阶段结论：**完成。** 当前主要问题已经不是 joint order、body mapping 或头部映射。

## 第 2 阶段：跑通 SONIC → X2 仿真闭环

建立完整的离线控制回路：

```text
X2 仿真状态
        ↓
SONIC/G1 policy
        ↓
29→31 语义适配
        ↓
X2 PD actuator
        ↓
MuJoCo / IsaacLab 下一状态
```

完成内容：

- X2 MJCF 可加载，官方站姿、PD 推进和状态回读均可运行。
- SONIC source checkpoint 可加载到 X2 环境。
- 1062 维观测到 31 维动作的训练、checkpoint、ONNX 导出入口跑通。
- 排除无地面导致 root 下沉的假故障，正式 smoke 使用 plane 地面。

阶段结论：**完成。** 闭环可以运行，但“能运行”不等于“已经会稳定跟踪”。

## 第 3 阶段：建立 X2 参考动作库

训练不能只靠站立 smoke，需要让真实人体动作变成 X2 可读取的 reference motion：

```text
AMASS / PHUMA / receiver 动作
        ↓
SMPL-X / GMR 重定向
        ↓
X2 root + 31DOF + joint order + fps
        ↓
SONIC MotionLib cache
```

完成内容：

- 接入 CMU、BML、ACCAD、HDM05、KIT、SFU、Transitions 等数据。
- 形成 363 条 X2 curriculum reference，并通过格式、FK、连续性门。
- 后续获得 PHUMA 风格的 clean291 物理筛选动作集。
- 明确 wrist pitch/roll 缺少真值时不伪造手腕数据。

阶段结论：**完成训练入口，数据质量仍有边界。** 数据数量已经不是主要短板；根部和足端的动力学一致性比继续堆数量更重要。

## 第 4 阶段：把 X2 执行器响应接入仿真

理想 PD 仿真和真实 X2 电机响应不同，因此采集并处理 readonly 日志：

```text
真机 q_cmd / q_state / dq / IMU
        ↓
辨识 delay + first-order response
        ↓
X2 nominal actuator domain
```

完成内容：

- Session02、Session03、Session04 被转换为 canonical 31DOF 数据。
- Session03+04 合计 88 段、约 4056 秒。
- 20/31 个关节得到可靠响应拟合；头部禁用。
- 建立 nominal 响应、随机化响应和 ideal 响应三类仿真配置。

阶段结论：**完成第一版。** 官方 PD 参数不是前置 blocker；现有日志足以建立训练用合理响应域，但仍不等价于精确数字孪生。

## 第 5 阶段：建立严格、可复现的晋级门

不能用单次 reward 或肉眼挑 checkpoint，因此建立固定评测：

```text
同一 seed + 同一 fixed16 motions + deterministic action
        ↓
nominal X2 actuator domain
        +
ideal dynamics off-domain
        ↓
root / body / joint / wrist / EE / foot / episode length / reward
```

规则：

- checkpoint 必须在固定条件下可重复。
- 同一个 checkpoint 必须同时检查目标域和理想动力学离域。
- 改善一项、明显牺牲另一项不能叫迁移成功。
- 所有候选只在仿真中评估，未通过门就不考虑真机。

阶段结论：**完成。** 后续大量“9/10”“局部变好但不晋级”都来自这套 fail-closed 规则。

## 第 6 阶段：按 Any2Any 建立 LoRA 训练方式

没有从零重训整个 SONIC，而是保留 G1 的动作先验，只适配目标机器人动力学：

```text
冻结：FSQ、encoder、kinematic decoder、source base weights
训练：actor dynamics-decoder LoRA、critic LoRA、action std
```

完成内容：

- LoRA smoke、保存、恢复、有效权重合并和 exact replay 通过。
- 修复学习率被自适应机制隐式放大的问题。
- 加入 source retention，避免小数据短训立即破坏原策略。
- 验证 LoRA 确实改变 X2 闭环，而不是 checkpoint 没有真正生效。

阶段结论：**完成。** Any2Any 式训练管线已经可信；后续问题是优化目标和闭环冲突，不是训练入口造假或未接通。

## 第 7 阶段：确认“短训下降”不等于路线错误

早期短训经常比 source baseline 差，因此进行了连续 500/1000 轮训练：

- 多条训练轨迹在前 100–200 轮下降，约 300 轮后出现恢复。
- 证明短训确实可能误判方向。
- 但恢复后的 checkpoint 仍在 root position 和 foot stability 之间交换性能。

阶段结论：**部分成功。** Any2Any dynamics adaptation 有效，但延长训练不会自动解决多目标冲突。

## 第 8 阶段：审计根部—足端物理一致性

对 363 条 reference 做逐帧 MuJoCo FK 和支撑相位审计，发现明显 foot skating、支撑漂移和 root/foot 不一致：

- 构造 clean67 低冲突子集。
- 接入 PHUMA 风格 clean291 数据。
- 尝试物理一致数据、课程训练和 checkpoint 插值。
- 更干净的数据能降低部分训练冲突，但不能单独解决高冲突动作上的泛化。

阶段结论：**诊断完成，数据路线局部有效。** 继续寻找更多同类 AMASS 数量价值较低；真正完整的 kinodynamic reference 仍会更好，但不是当前才能继续工作的硬性依赖。

## 第 9 阶段：尝试结构分离与稳健干预

参考 XHugWBC、GeoRA/KDMR 等思路，尝试把上肢跟踪和下肢平衡解耦：

- 下肢-only、hip-yaw、腰/髋/膝/踝分组 LoRA。
- root-risk residual gate、row mask、retention、插值和 cause-specific sampling。
- 独立 EE/foot termination penalty 和连续风险。
- 对 foot/orientation 分组求梯度，确认多个输出行存在稳定梯度冲突。

曾出现 nominal 10/10，但被证明依赖 float32 舍入边缘，且 response-off 只有 5/10，因此按稳健性规则否决。

阶段结论：**获得原因证据，没有可靠晋级模型。** 瓶颈不是某一个关节行，而是闭环全身动力学耦合。

## 第 10 阶段：把 root、腰、腕和足端的因果关系拆开

这一阶段不再盲目扫超参数，而是做结构因果审计：

- 分开统计 root、waist、torso、左右 wrist、左右 foot。
- 交换 dynamics decoder 的 7 层 LoRA 与最终 31 维输出层。
- 发现参考 waist pitch 基本为 0、waist roll 很小，但策略会把腰输出推向关节限位。
- 确认 root 收益主要来自共享隐藏 LoRA，而不是某个最终腰部输出行。

阶段结论：**完成。** “只修一个腰部 action”不能局部解决问题，因为腰、根、脚和腕通过闭环动力学相互影响。

## 第 11 阶段：验证局部腰部/输出层修复是否可行

进行了两条正式路线：

- 11A：完整 dynamics-decoder LoRA + 腰部参考奖励，连续 300 轮。
- 11B：只训练 waist pitch/roll 两个最终输出行，并测试残差插值。

腰部、左腕或 root orientation 能局部改善，但总会伴随 root position、右腕、EE 或 foot 回归。

阶段结论：**阶段完成，候选不晋级。** 证明问题不能靠继续缩小 LoRA 层数或再调一个标量 reward 解决。

## 第 12 阶段：约束式 Any2Any，得到当前综合最佳

把多个风险从互相竞争的单一 reward 改为显式物理约束：

```text
root position
root orientation
left/right wrist
foot
waist reference
```

关键变化：

- 使用 checkpoint 权重锚，保护原有能力。
- 使用拉格朗日约束，不只依赖 reward 加权。
- 从全局均值逐步升级到 p95/tail 风险。
- 最终按每个 motion 独立计算 p95 和最差 20% 帧，避免好动作掩盖坏动作。

Stage12D-100 相比保守 baseline：

- nominal reward：`0.4660 → 0.5761`
- ideal reward：`0.6809 → 0.7352`
- nominal EE termination：`64.95% → 61.61%`
- ideal EE termination：`49.32% → 43.28%`
- root 综合改善或基本守平
- 但 foot termination 在双域仍有回归

阶段结论：**明确改善，但未全门通过。** Stage12D-100 是目前综合最佳 checkpoint，不是真机部署模型。

## 第 13 阶段：连续训练 3000 轮，检验长期趋势

从 Stage12D-100 继续训练：

```text
64 env × 24 step × 3000 iteration
= 4,608,000 simulator transitions
```

结果：

- 训练完整结束，无 NaN/Inf，checkpoint 和 replay 边界正常。
- fixed reward 在后期恢复，证明“短训先跌、长训可能回升”部分成立。
- Stage13-3000 的腕/腰和 reward 更好。
- 但 root orientation、joint tracking 和 foot stability 明显变差。
- root-orientation 拉格朗日乘子从第 1551 轮起长期饱和，现有约束已压不住 reward 梯度。

阶段结论：**实验完成，模型不晋级。** Stage13-500 是该阶段最均衡的视觉对照点；综合最佳仍是 Stage12D-100。原配置直接延长到 8000 轮价值很低。

## 第 14 阶段：修正约束机制，并完成统一视觉复核（下一阶段）

### 目标

不是马上再跑一次大长训，而是先让训练目标真正约束住 root orientation 和逐动作 foot 尾部，同时确认量化指标对应的真实仿真表现。

### 任务清单

- [ ] 渲染 baseline、Stage12D-100、Stage13-500、Stage13-3000 的同动作 MuJoCo 对照。
- [ ] 核对 root orientation/foot termination 是否对应肉眼可见的失衡、打滑或错误姿态。
- [ ] 对 root-orientation cost 做尺度归一化、自适应乘子上限或更新率修正，解除长期饱和。
- [ ] 将 foot 与 root-orientation 的 per-motion p95/最差帧约束直接放进训练，而不是只看混合 batch 均值。
- [ ] 让训练期风险统计与 fixed16 晋级指标使用同一语义，避免“训练约束显示合格、固定动作却恶化”。
- [ ] 从 Stage12D-100 做受控 100–300 轮验证，不从 Stage13-3000 继续。

### 阶段成功条件

同一 checkpoint 至少做到：

- root orientation 不差于 Stage12D-100；
- foot termination 不继续恶化，并尽量回到保守 baseline 附近；
- 保留 Stage12D 已获得的 reward、root 和 EE 收益；
- nominal 与 ideal 两域方向一致；
- 量化指标与 MuJoCo 视频判断一致。

## 第 15 阶段：新约束配方的正式长训与最终仿真晋级（后续阶段）

### 前置条件

只有阶段 14 的短程因果验证通过，才启动这一阶段；否则不继续用算力证明同一个失败方向。

### 任务清单

- [ ] 用阶段 14 的新配方连续训练 3000 轮，并按固定间隔保存 checkpoint。
- [ ] 如果 3000 轮趋势仍稳定，再决定是否延长到 Any2Any 量级的 8000 轮。
- [ ] 对所有关键 checkpoint 做 nominal + ideal 双域 fixed16 评估。
- [ ] 对 clean291/更广 motion panel 做逐动作、逐失败原因评估，防止 fixed16 过拟合。
- [ ] 统一比较 baseline、Stage12D、Stage14 候选和长训候选的视频与指标。
- [ ] 只导出真正通过综合门的 ONNX 候选；未通过则保留 Stage12D，不伪造“最终模型”。

### 阶段成功条件

最终候选必须同时满足：

```text
比原保守 baseline 明确更好
        +
不靠牺牲 root/foot 换取 wrist/EE
        +
nominal 与 ideal 双域均成立
        +
更广动作集没有明显系统性回归
        +
MuJoCo 视觉表现确实是稳定、合理的 X2 动作跟踪
```

阶段 15 通过后，才进入后续“11 IMU 智能服装 → reference/command → X2 SONIC”的全链路接入与延迟测试；真机安全部署仍是再下一关，不包含在阶段 15 内。

## 当前总状态

```text
迁移接口与仿真地基       ██████████ 100%
数据与执行器响应入口     ██████████ 100%
Any2Any 训练与评估工具   ██████████ 100%
获得优于保守 baseline 的模型 ████████░░ 80%
获得全面稳定的仿真晋级模型   ██████░░░░ 60%
智能服装到 X2 全链路验证     ███░░░░░░░ 30%
真机部署资格                 ░░░░░░░░░░  0%
```

当前一句话：**我们已经证明 SONIC/G1 能迁移出比原保守 baseline 更好的 X2 模型，但还没有解决“上肢收益与 root/foot 稳定性互换”的最后核心冲突。**
