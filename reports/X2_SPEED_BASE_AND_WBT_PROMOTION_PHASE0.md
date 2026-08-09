# X2 速度保底后端与 WBT 晋升：Phase0 总报告

日期：2026-08-09
状态：第一批 1–9 项完成；第 10 项“等待用户审阅”已触发。
本轮边界：未长训、未启动 PPO、未向真机发送命令、未升级 SDK、未修改源数据或旧 checkpoint。

## 一句话裁决

速度保底线已经能被准确冻结，但还没有统一通过复合 81-case；WBT 的官方模型与数据/训练契约已重新建立，不过三条严格 A/B 证明旧 GMR 机器人 MJCF 与 AimDK v1.0 官方 MJCF 在机器人数值 contract 上本来就相同，因此“换官方模型文件即可修复 reference”的假设被否定，下一步必须先改 **IK/body/contact contract**，不能直接全量重定向或长训。

## Phase0 任务完成表

| # | 任务 | 状态 | 核心结果 |
| ---: | --- | --- | --- |
| 1 | 冻结 BASE_LOCOMOTION / BASE_TRANSITION | ✅ | Stage219/250 与 Stage306 角色分离，权重/ONNX/契约/哈希已记录 |
| 2 | 汇总速度线门禁覆盖 | ✅ | 没有同一候选、同一部署配置完整覆盖并通过复合矩阵 |
| 3 | 官方 joint/body/contact/mirror contract | ✅ | 31 MJCF / 29 body / 15 lower-waist 边界清楚；FK/mirror 测试通过 |
| 4 | 盘点 AMASS/PHUMA/BONES | ✅ | 展开源仍有 AMASS 35.748h、PHUMA 21,731、BONES 370；原压缩包缺失 |
| 5 | 固定 20–30 条诊断动作 | ✅ | 24 条：AMASS 12 / PHUMA 7 / BONES 5；train candidate 14 / held-out 10 |
| 6 | 三条旧/官方模型 A/B retarget | ✅ 否证 | upper/walk/squat 全字段最大差 `0.0` |
| 7 | Bronze/Silver/Gold 门禁草案 | ✅ | 运动学、接触、官方动态三层不再混写 |
| 8 | 忠实 Any2Any 预注册计划 | ✅ | SONIC-specific `g1_dyn + critic` 为主基线，S7 只作后续单变量消融 |
| 9 | 用户/智元确认问题 | ✅ | 控制权、watchdog、真机 command/state contract 为 P0/P1 |
| 10 | 用户审阅后再批量/训练 | ⏸ | 本报告即审阅点；当前不自动继续 |

## Track A：速度型保底线

### 冻结资产

- `BASE_LOCOMOTION`：Stage219 `model_2600.pt` / `stage219_s2600_actor.onnx`，部署语义采用 Stage250 修复；
- `BASE_TRANSITION`：Stage306 `model_2652.pt` / `stage306_s2652_transition_head_actor.onnx`；
- 两者都依赖 stand backend、15DoF gait template、官方 MuJoCo adapter 和精确 action clipping/last-action contract，不能只保存一个 ONNX。

关键纠错：Stage250 不是新 checkpoint；现有门禁所谓 `official_kp_ankle` 也不是完整官方逐关节 PD，而是 ankle `40/20`、其余 lower/waist `300/20` 的项目 profile。Stage306 trace 中“93D observation”是报告字段错误，真实 ONNX 输入为 123D。

### 能力与缺口

- Stage219/250 nominal 基础运动曾达到 24/24；
- Stage306 提供 Future-intent/transition 能力，但 nominal turn 仍为 7/8、stiff-fixed 为 3/5，不能替代 BASE_LOCOMOTION；
- 当前不存在一个相同 checkpoint + 相同 supervisor + 相同部署配置完整覆盖 `PD×upper×motion×seed` 的通过矩阵；
- 所以 BASE 是可靠回退和对照，不是“完整鲁棒后端完成”。

### A1 已有前置资产（非本轮训练）

在任务卡切换前已经从 Stage326 stiff-fixed 的 5 条官方 trace 提取了 90 个健康停车边界状态：63 个最终通过、27 个最终失败。它们是 recovery reset-state curriculum，不是 motion reference，也没有启动 recovery 训练。本轮保留提取脚本、测试和哈希，避免丢失，但不把它写成 A1 已完成。

## Track B：官方 X2 WBT

### Canonical contract

- 官方 MJCF：31 个执行关节；
- 官方 sample RL：29 个 body action，仅排除 head yaw/pitch；
- 速度保底后端：15 个 lower/waist action；
- WBT 首阶段：29DoF body，头锁定，不涉及灵巧手；
- feet/hands tracking proxy：ankle-roll / wrist-roll link；官方模型没有独立 palm/sole tracking site；
- 每脚 12 个半径 5 mm 接触球，左右 permutation 已显式冻结。

镜像 joint round-trip 为 `0`；contact point 为 `0`。body FK 最大残差 `2.105 mm`、rotation matrix element `0.005888`，来自官方左右 CAD 的毫米级非对称，因此预注册为 3 mm / 0.007 几何容差，未伪造成 exact-zero。

### 数据资产

- AMASS：11,798 条，11,796 可读，35.748 h；
- PHUMA：21,731 条 G1 29DoF NPY；
- BONES：370 条已抽取 BVH，旧漏斗实际处理 344；原 `soma_uniform.tar.gz` 已不在；
- 24 条固定诊断面板的 source path 与 SHA-256 全部验证存在；
- 缺口仍包括：真实 X2 GRF/COP/centroidal momentum、BONES 完整 archive、旧 cache 与 AimDK v1.0 contract 的绑定。

### 最重要的 A/B 结果

同一 AMASS upper-only、walk、squat 在以下条件下分别用旧 X2 robot XML 和 AimDK v1.0 官方 `x2.xml` **重新运行**：30 Hz、full root、GMR root-z、smooth9、同一 v4 IK、同一 solver/damping，无 postprocess。

结果：dof、root xyz、root quaternion、pose-aa 的全局最大差均为 `0.0`。

进一步模型审计表明：两个 robot XML 的 joint/body 名称、关节位置/轴/限位、质量、惯量、damping、armature 与 actuator 参数都数值相同；旧 XML 只额外内置一个 `contype=0/conaffinity=0` 的 floor 及可视化资源。于是：

> 早期失败不能再归因于“GMR 使用了错误的 X2 机器人本体文件”。真正仍未解决的是旧 v4 IK 的 link/offset/orientation/scale objective、contact grounding 语义，以及运动学轨迹的动态可执行性。

这也是为什么当前不能因为“引入官方文件”就全量重跑 35.7 h AMASS：那会得到数值相同的旧问题。

## 分级门禁

- Bronze：有限、连续、关节/地面安全、FK 与左右语义正确；
- Silver：至少一个完整 DS→SS→DS，stance slip/excursion、摆脚 clearance 与 contact timing 过门；
- Gold：在 AimDK 官方 MuJoCo 中完整存活，root/body/contact/action/effort 同时过门，并保留踢腿、弓步、深蹲、转向事件语义。

静态 upper-only 可以作为 Bronze 保持数据，但不会因“双脚没滑”被叫成 locomotion Silver。任何 FK/contact/COM 量都标记为模型估计，不冒充真实足底力/COP。

## 忠实 Any2Any 基线锁定

Any2Any 对 SONIC 的明确说明是 dynamics decoder + critic LoRA，FSQ 与其他预训练部分冻结；Figure 7 S7 的更广范围是 Oli-WBT→Luna 消融结果。因此：

1. 首个 baseline 训练完整 SONIC `g1_dyn` 与 critic 的 Linear LoRA；
2. 冻结 motion/reference encoder、FSQ、kinematic decoder、dense source weights、action std；
3. 不混入 contact preview、response adapter、CEM、CWI 或 recovery switching；
4. 先跑 0/5/25 iterations，之后按 200/1000/8000 分级解锁；
5. held-out 10 条禁止参与训练、阈值、early stop 与 checkpoint selection；
6. 每次同时保留 frozen-aligned、full FT、faithful LoRA、from-scratch 和基础 locomotion 对照。

这意味着过去 X2 工作的主要问题不是“只给 decoder 末层加 LoRA”；过去完整 `g1_dyn + critic` 范围已大体符合 SONIC-specific 描述。更大的偏差是训练长期围绕两条短动作及其镜像，以及 reference 质量没有达到动态 Silver。

## 现在不允许做什么

- 不全量重定向三个数据源；
- 不启动 WBT PPO 或 8000-iteration 长训；
- 不继续盲扫 LoRA、reward 或 checkpoint；
- 不把速度 backend 的 walk 能力写成 WBT；
- 不升级真机 v1.0，不向真机发命令；
- 不把官方 MuJoCo 通过等同于真机接口可用。

## 用户审阅后，最小高价值下一步

由于“只换 MJCF”已经被否证，下一轮仍属于数据准备，不属于自定义 policy 创新：

1. 对固定三条/随后 24 条做 **v4 IK vs canonical body/contact contract** 单变量 A/B；
2. 首先替换无几何依据的 foot/knee/wrist offset 与 tracking proxy，而不是动 root/reward/LoRA；
3. 以官方 12 点足底几何重新计算 ground/contact 标签；
4. 只有 Bronze/Silver 指标跨动作改善，才扩到全量 AMASS→PHUMA→BONES；
5. 只有形成独立 official-X2 Silver/Gold train 与 held-out，才启动忠实 Any2Any 0/5/25 smoke。

如果 canonical IK/contact A/B 仍不能产生 Silver，停止继续堆数据，转向 contact-constrained/kinodynamic retarget teacher；不能靠长训替 reference 补物理。

## 主要证据

- [速度后端冻结清单](baseline/x2_speed_backend_manifest.md)
- [速度门禁覆盖](baseline/x2_speed_gate_coverage.md)
- [官方模型 contract](retarget/x2_official_model_contract.md)
- [旧数据盘点](retarget/x2_wbt_legacy_data_inventory.md)
- [24 条诊断面板](retarget/x2_wbt_diagnostic_panel.md)
- [三条旧/官方 A/B](retarget/x2_official_retarget_smoke3.md)
- [分级门禁](retarget/x2_wbt_tier_gates.md)
- [忠实 Any2Any 计划](retarget/x2_faithful_any2any_phase0_plan.md)
- [厂商确认问题](deployment/x2_vendor_confirmation_questions.md)

## Phase0 结论

本轮到达的甜点位不是“训练准备 100%”，而是把两条路线的事实边界和一个关键误判排清：BASE 可冻结但复合鲁棒性未完成；WBT 具备可审计的官方 contract 和诊断面板，但模型文件替换本身无效。下一步价值集中在 canonical IK/contact A/B，而不是更长训练。
