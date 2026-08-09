# X2 速度型保底后端冻结清单

版本：Phase0-v1
日期：2026-08-09
范围：只冻结速度保底线；不代表 Whole-Body Tracking、真机或完整鲁棒矩阵已通过。

## 冻结裁决

当前必须保留两个不同角色，禁止合并成一个“最佳模型”：

1. `BASE_LOCOMOTION`：Stage219 `model_2600.pt` / `stage219_s2600_actor.onnx`，使用 Stage250 修正后的部署语义。Stage250 **不是新 checkpoint**，而是同一 Stage219 权重的 action clipping、last-action feedback 与转向部署契约修复。
2. `BASE_TRANSITION`：Stage306 `model_2652.pt` / `stage306_s2652_transition_head_actor.onnx`。它增加 Future-intent/response/transition 能力，但尚未覆盖或替代 `BASE_LOCOMOTION` 的全部航向与鲁棒性能力。

同样重要：目前可运行的“后端”不是单个 ONNX。官方门禁还依赖 stand backend、15DoF gait template、官方 MuJoCo 适配器和方向特定 supervisor。任何复现都必须冻结这些依赖。

## 共同官方仿真域

- 官方包：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/downloads/aimdk-x2-v1.0.0-official.zip`，SHA-256 `5bbcf724d54fb28f153db0d272f9acb7906bb1d2cac7dd7ccdc699a5c7eeab35`。
- 官方运行时：AimRTE `v0.8.18`、interface `v0.9.0.7`；随包 MuJoCo 动态库为 `libmujoco.so.3.3.7`，SHA-256 `9bfb4d37d1182338eaf24efdf8390e718e4f1b8597a3674fde3884dde62ff97f`。
- X2 MJCF：`.../configuration/robot/lx2501_3_t2d5/model_info/x2.xml`，SHA-256 `3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3`。
- scene：同目录 `scene.xml`，SHA-256 `7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63`。
- 官方门禁镜像：`x2-aimdk-humble:1.0`；控制器通过 ROS2 与官方模拟器闭环，actor 固定 `50 Hz`（`dt=0.02 s`），仿真状态话题为 500/1000 Hz。
- 证据：`x2_rl_deploy_mujoco/version.json`、`x2_rl_deploy_mujoco/lib/libmujoco.so.3.3.7`、`x2_rl_deploy/README.md`、`tools/official_x2/run_official_gate_case.sh`、`tools/official_x2/run_official_gate_inner.sh`。

## 共同控制契约

### 15DoF action order

`left hip pitch/roll/yaw, left knee, left ankle pitch/roll, right hip pitch/roll/yaw, right knee, right ankle pitch/roll, waist yaw/pitch/roll`。

action scale（rad）：`[0.4,0.4,0.4,0.4,0.12,0.08, 0.4,0.4,0.4,0.4,0.12,0.08, 0.4,0.16,0.16]`。

### gait template

- 路径：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz`
- SHA-256：`16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d`
- 结构：40 帧、周期 `0.8 s`、双支撑比例 `0.30`、15DoF；执行时 template scale=`0.15`。
- 注意：这是项目生成的 X2 gait template，不是 SDK 自带策略文件。

### clip / last-action

冻结执行顺序：

```text
ONNX actor mean
→ clip[-1,1]（该值写入下一帧 previous/last action）
→ 加 0.15 × gait template / action_scale
→ execution clip[-1,1]
→ default pose + action_scale × action
→ position PD target
```

部署适配器：`tools/official_x2/stage208_official_mujoco_adapter.py`，当前文件 SHA-256 `72ff517b9105846570bab89858d8613e571890240840a7a2be9543c5214a14e1`。Stage250 证实旧部署把未裁剪 ONNX mean（最大 `6.26`）回灌到 last-action 会造成闭环自激；该旧语义不得再使用。证据：`reports/official_x2/stage250_rsl_action_contract_fix_20260808.{md,json}`。

### 实际门禁 PD（必须按代码而不是简称理解）

当前通过结果使用 `PD_PROFILE=official_kp_ankle`，它并非“完整官方逐关节 PD”：

- 四个 ankle pitch/roll：`Kp=40, Kd=20`；
- 其余 11 个 lower/waist：`Kp=300, Kd=20`；
- upper shoulder/elbow：`40/5`，wrist：`30/3`，head：`50/5`；
- `soft/nominal/stiff` 只把 lower/waist 的 Kp/Kd 乘以 `0.9/1.0/1.2`，不乘 upper/head。

门禁默认姿态也是 `stage208` profile，而不是 SDK `official_v1` 深蹲姿态。以上由 `stage208_official_mujoco_adapter.py::_pd/default_pose` 与 `run_official_gate_inner.sh` 默认参数直接证明。任务卡中“官方逐关节 PD”属于宽泛描述，复现时必须采用这里的精确定义。

## BASE_LOCOMOTION

### 冻结资产

- 训练 checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt`
- checkpoint SHA-256：`abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb`
- ONNX：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage219_s2600_actor.onnx`
- ONNX SHA-256：`b95bad3680658c7c25be50f236f070c80b7ff7ba8992355cec2ddfb1ee53c0f9`
- ONNX 校验：PyTorch/ONNX 最大绝对误差 `2.3842e-7`；证据为同目录 `stage219_s2600_actor.export.json`。
- 隐式辅助 stand backend：`.../models/stand_backend_scratch_i150_actor.onnx`，SHA-256 `edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565`。门禁 runner 默认装载它；因此不能只归档 locomotion ONNX。

### Observation schema（93 float32）

`base linear velocity(3) + base angular velocity(3) + projected gravity(3) + command[vx,vy,wz](3) + 31DoF q-default(31) + 31DoF dq(31) + previous clipped 15D actor action(15) + gait/contact phase(4)`。

### Action schema（15 float32）

输出是上述 lower/waist 15DoF 的归一化 PD residual，不是 torque，也不是全 31DoF target。上肢 14DoF 和头 2DoF 由明确的独立 target 路径持有。

### 已证实能力（各报告独立，不合并成一个新矩阵）

- Stage250 nominal、低/中速、straight/right/left：`24/24`。证据：`reports/official_x2/stage250_rsl_action_contract_fix_20260808.{md,json}`。
- Stage251 nominal straight × fixed/slow/fast upper：`9/9`。证据：`reports/official_x2/stage251_upper_straight_panel.{md,json}`。
- Stage252 nominal turn：fixed/right/left 与 slow/right/left 全过，fast-right `3/3`，fast-left `2/3`，合计 `17/18`。证据：`reports/official_x2/stage252_upper_turn_panel.{md,json}`。
- Stage253 只把 left yaw progress gain 从 2.0 改为 2.5，left × fixed/slow/fast=`9/9`；它不是一套重新运行的左右联合矩阵。证据：`reports/official_x2/stage253_left_yaw_gain_panel.{md,json}`。
- Stage257 straight × PD `{0.9,1.0,1.2}` × upper `{fixed,fast}`=`14/18`；不是完整通过。证据：`reports/official_x2/stage257_pd_upper_straight_panel.{md,json}`。
- Stage263 使用另一套横向 supervisor 后仍为 `14/18`，失败单元发生迁移，不能与 Stage257 互补拼成通过。证据：`reports/official_x2/stage263_pd_upper_straight_panel.{md,json}`。

### 已知失败 / 边界

- 没有同一原子评估运行覆盖 `PD 3 × upper 3 × motion 3 × seed 3`。
- PD 复合域只测 straight 和 fixed/fast；没有 slow，也没有 soft/stiff 左右转。
- Stage257 的 soft-fast=`0/3`、stiff-fast=`2/3`；Stage263 的 soft-fixed=`2/3`、stiff-fixed=`0/3`。两套 supervisor 都不能晋级。
- 官方闭环尚无被接受的显式 delay/noise 门；Stage224/225 的 direct-MJCF 探索文件是未跟踪、不同闭环语义的诊断，不能记入冻结候选通过率。
- 不是 Whole-Body Tracking；不能表达指定脚位、弓步、踢腿或任意人体下肢姿态。
- 官方 MuJoCo 通过不等价于真机，SDK README 明确说明真机需重新参数适配。

## BASE_TRANSITION

### 冻结资产

- checkpoint：`/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_11-18-41_stage306_s2642_transition_head_h1p0_long15_lr1e4_v1/model_2652.pt`
- checkpoint SHA-256：`fa2cce83d3ec58c41d16f31bb5a627bdeeb80ea8dba86238301ba10a998a5365`
- ONNX：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage306_s2652_transition_head_actor.onnx`
- ONNX SHA-256：`da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c`
- ONNX 校验：PyTorch/ONNX 最大绝对误差 `1.1176e-7`；证据为同目录 `stage306_s2652_transition_head_actor.export.json`。
- 同样依赖 gait template、adapter 和 stand backend；建议冻结的 nominal 部署甜点还包含：future-stop preview=`0.5 s`、split locomotion intent、brake gain=`1.5`、brake→stand blend=`1.0 s`、event hold 最少 `0.5 s`，straight 使用 lateral recovery supervisor，turn 使用 right gain=`3.0` / left gain=`2.5`。

### Observation schema（123 float32）

`BASE_LOCOMOTION 93D + upper current delta(14) + upper future-minus-current delta at 0.6 s(14) + normalized current locomotion intent vx(1) + normalized future-minus-current vx(1)`。

transition head 实际读取 2D locomotion intent + 4D gait phase；response branch 读取 lower q/dq/previous action、locomotion intent 和 gait phase。基础 actor 冻结，残差被限制在低维腰腿协调模态。注意：原始 trace JSON 把 observation_contract 固定写成“93D”，这是报告字段 bug；实际 ONNX 输入宽度以 export manifest 的 123D 为准。

### Action schema（15 float32）

与 BASE_LOCOMOTION 相同的 15DoF normalized lower/waist PD residual。transition coordination output 上限 `0.10`，response branch 上限 `0.05`；零前向命令严格回退基础 actor（export manifest 误差 `0.0`）。

### 已证实能力（部署配置不得跨报告混拼）

- Stage314 split-intent、nominal straight：fixed `3/3`、fast `3/3`。
- Stage321 future preview 0.5 s、nominal turn：fixed right/left=`4/4`，fast right=`2/2`，fast left=`1/2`，合计 `7/8`。证据：`reports/official_x2/stage321_future_stop_preview0p5_panel.json`。
- Stage325 同一 0.5 s preview 的 nominal straight 回归：fixed `3/3`、fast `3/3`。证据：`reports/official_x2/stage325_straight_preview_regression.json`。
- Stage326 同一模型进入 stiff 1.2× straight：fixed `3/5`、fast `5/5`。证据：`reports/official_x2/OFFICIAL_X2_STAGE316_328_FUTURE_PREVIEW_AND_STIFF_AUDIT_20260808.md` 与对应 raw trace。

### 已知失败 / 边界

- nominal fast-left 仍有概率性欠转；Stage321 为 `1/2`。
- stiff-fixed 为 `3/5`，失败集中在 move→stop 后掉出站立/恢复吸引域；不能用 stiff-fast `5/5` 替代 fixed 失败。
- 无 soft 0.9×结果；无 slow-upper结果；无 soft/stiff turn 结果；没有完整 81-run 原子矩阵。
- Stage329 zero-upper 课程独立复验 fixed=`3/5`、fast=`4/5`；Stage332/333 upper-activity 条件化最高仅 `2/3`，均未晋级。证据：`reports/official_x2/OFFICIAL_X2_STAGE329_333_ZERO_UPPER_AUDIT_20260808.md`。
- Stage306 不能覆盖 BASE_LOCOMOTION：其名义转向只有 `7/8`，且其 stiff 能力不稳定。

## 统一门禁定义

`full_gate_pass` 至少同时要求 startup、move、stop 子门通过。当前适配器的关键固定阈值包括：startup root-z≥`0.60 m`、tilt≤`0.30 rad`、前进≥`0.10 m`、反向窜动≤`0.03 m`；move root-z≥`0.45 m`、tilt≤`0.40 rad`、straight 前进≥`0.50 m`、横漂≤`0.30 m`、航向偏差≤`0.30 rad`，turn yaw ratio∈`[0.50,1.50]`；stop root-z≥`0.45 m`、tilt≤`0.30 rad`、漂移≤`0.15 m`、尾段速度≤`0.03 m/s`。权威实现：`tools/official_x2/stage208_official_mujoco_adapter.py`。

## Phase0 最终判断

- `BASE_LOCOMOTION` 已冻结为可靠 nominal 安全回退，但不是复合鲁棒后端。
- `BASE_TRANSITION` 已冻结为 start/stop 与 Future-intent 研究基线，但不能替换 `BASE_LOCOMOTION`。
- 当前不存在一个能诚实标记为“完整矩阵通过”的候选。
- 下一项高价值工作仍是独立 stop-to-stand/recovery 后端，并从真实成功、临界、失败停车末态初始化；在此之前不应继续扩大 transition adapter 或宣称长训解锁。
