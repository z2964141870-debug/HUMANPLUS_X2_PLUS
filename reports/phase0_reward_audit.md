# Phase 0 Reward 审计（Stage152-B）

日期：2026-07-26  
来源：config.yaml（SHA256 见 manifests/baseline_manifest.yaml）+ training_diagnostics.jsonl（200 条）交叉确认。

## 1. Active reward terms（权重非零，运行时 Episode_Reward 非零）

| # | term | weight | 作用范围 | 首步 episode reward | dual 分组候选 |
|---|---|---|---|---|---|
| 1 | tracking_anchor_pos | 0.2 | base_link anchor 位置 | 0.00226 | **loco** |
| 2 | tracking_anchor_ori | 0.5 | anchor 朝向 | 0.00559 | **loco** |
| 3 | tracking_relative_body_pos | 1.0 | 全身 14 body（anchor 相对，stance 缩放） | 0.01122 | ⚠ 混合 |
| 4 | tracking_relative_body_ori | 1.0 | 全身 14 body | 0.00959 | ⚠ 混合 |
| 5 | tracking_body_linvel | 1.0 | 全身 body 线速度 | 0.00953 | ⚠ 混合 |
| 6 | tracking_body_angvel | 1.0 | 全身 body 角速度 | 0.00635 | ⚠ 混合 |
| 7 | tracking_vr_5point_local | **2.0**（最大） | 实为 3 点：头点+双腕（见 §3.2） | 0.01917 | **upper** |
| 8 | tracking_foot_contact_phase | 1.0 | 双足接触/载荷转移 | -0.00483 | **loco** |
| 9 | tracking_reference_swing_foot_height | 1.0 | 摆动足高度 | 0.00135 | **loco** |
| 10 | feet_acc | -2.5e-06 | ankle 关节加速度 | -0.00872 | **loco** |
| 11 | undesired_contacts | -0.1 | 非足/腕/肘 body 接触 | -0.00680 | **loco**（安全） |
| 12 | joint_limit | -10.0 | 全部关节 | -0.00182 | ⚠ 混合（安全正则） |
| 13 | action_rate_l2 | -0.1 | 全部动作 | -0.01003 | ⚠ 混合（平滑正则） |
| 14 | anti_shake_ang_vel | -0.005 | 腕 link 角速度 | -0.00018 | **upper** |

## 2. Inactive terms（weight=0，15 个）

tracking_anchor_linvel、anchor_path_error_penalty、tracking_selected_joint_pos、
tracking_ee_height、tracking_reference_stance_foot_slip、tracking_interfoot_relative_pos、
tracking_single_support_com、motion_completion_bonus、is_terminated、
ee_termination_penalty、foot_termination_penalty、ee_body_pos_risk、foot_body_pos_risk、
anchor_pos_termination_risk、anchor_ori_termination_risk。

不进入 contract 分组；后续任何阶段启用须重过 contract 门（见 DECISIONS.md D-002）。

## 3. 函数级审计结论（2026-07-26 补充，源码已读）

### 3.1 路线判定：只能走 term 级分组（路线 A）

`tracking_relative_body_pos/ori`、`tracking_body_linvel/angvel`、`tracking_vr_5point_local`
全部为 `exp(-mean_over_bodies(err)/std²)` 形式（rewards.py:630, 686ff, 1300ff, 1372ff），
对 body 聚合**非线性** → body 级拆分无法满足 ≤1e-6 等价，路线 B 否决。

term 级分组的精确等价挂钩点已确认：IsaacLab `RewardManager.compute()`
（reward_manager.py:129-159）线性累加 `term*weight*dt`，且 `_step_reward` 保留
`(num_envs, num_terms)` 逐 term 值（line 156）。按 term 子集求和 = 分组 reward，
与 scalar 差异仅为浮点加法顺序（≪1e-6）。**Contract 门可满足。**

### 3.2 关键语义修正：vr_5point 实为纯上肢 3 点项

本配置 `reward_point_body = [torso_link(+0.5m z 偏移≈头), left_wrist_roll_link,
right_wrist_roll_link]`（config.yaml:751-764），只有 3 个点、不含足；
`single_support_stance_foot_scale=1.0` 不触发足点分支（rewards.py:575 条件不成立）。
→ **权重最大（2.0）的 term 是纯上肢遥操作端点跟踪项，干净归入 V_upper。**

### 3.3 Dual 分组候选（两个预案，Phase 1 用 fixed rollout 统计定稿并预注册）

**G-A（端点语义，保守）**
- V_upper：tracking_vr_5point_local、anti_shake_ang_vel
- V_loco：其余 12 项（含 4 个全身模仿项、joint_limit、action_rate_l2）

**G-B（模仿并入 upper）**
- V_upper：G-A 的 2 项 + tracking_relative_body_pos/ori + tracking_body_linvel/angvel
- V_loco：anchor_pos/ori、foot_contact_phase、swing_foot_height、feet_acc、
  undesired_contacts、joint_limit、action_rate_l2

取舍依据（Phase 1 在 fixed rollout 上计算后决定）：
4 个全身模仿项与 loco 组/upper 组各自的逐步 reward 相关系数与方差占比；
决策在任何训练开始前预注册，禁止事后改组。
G-B 更贴近 CWI 的 task/style 划分（triple 时这 4 项再剥离为 V_imitation_style）；
G-A 的 upper 组信号更纯但方差占比小（vr_5point 均值 0.019 vs loco 组合计 ~0.02）。

**N1 负控制**（Phase 1 预注册固定 seed 生成）：把 14 个 active term 随机划成
与 E1 所选分组同样大小的两组（约束：两组各含至少一个 tracking 项），
种子与分组结果写入 manifest 后冻结。

## 4. 待办（更新）

- [x] 读 reward 函数源码，判定 body 聚合线性性 → 全部 exp-of-mean，路线 A 定案；
- [x] vr_5point 语义确认 → 纯上肢 3 点；
- [ ] Phase 1：fixed rollout 上统计各 term/组均值、方差、相关系数 → 定稿 G-A vs G-B；
- [ ] Phase 1：实现 vector reward（挂钩 RewardManager._step_reward）+ 等价性测试；
- [ ] Phase 1：预注册 N1 分组。
