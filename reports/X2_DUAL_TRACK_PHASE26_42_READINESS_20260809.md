# X2 双路线长训 Readiness 审计：BASE Phase26 / WBT Phase42

日期：2026-08-09
官方域：AimDK v1.0 MuJoCo（明确区分 closed ROS 与同进程 test-only MJCF）

## 一句话裁决

**当前没有一条路线满足长训解锁条件；保留 Stage250 nominal 基础行走能力，但停止继续扩大现有 recovery CEM、局部 reference repair 或 Any2Any PPO。**

这不是把项目判为“不可能”，而是证明现有训练入口的数据与 teacher 合同不成立；继续长训只会把计算量放大，不能修复缺失的动力学可行性。

## 本节点覆盖的 10 个实质任务

自 Phase36 大文件备份后：

1. WBT Phase37：official collision contact 全轨迹修复；
2. WBT Phase38：SLSQP 硬约束局部证书；
3. WBT Phase39：Sequential-QP 局部证书；
4. WBT Phase40：释放 root XY 的结构 A/B；
5. BASE Phase24：完整物理支持域交权；
6. BASE Phase25：role-aware 最早偏离审计；
7. WBT Phase41：centroidal / GRF 六维平衡可行性；
8. BASE Phase26：同快照 dynamic bridge CEM；
9. WBT Phase42：contact-first centroidal teacher；
10. 本 Readiness 审计、测试、版本与大文件封版。

## BASE 路线

### 保留下来的能力

- Stage219 checkpoint 按 Stage250 部署合同，在 official nominal 条件下保留站立、起步、前行与当时三条 full-gate 通过记录。
- 对应 checkpoint、ONNX、MP4/GIF 均未被 Phase21–26 覆盖。
- 这只证明冻结 nominal envelope；肉眼可见后仰、严格 fixed-upper / stiff matched-event 的横漂、航向和停车问题仍然存在，不能写成稳健原生 locomotion。

### Phase24–25 已排除的解释

- 等待 `previous action + generator DS + base angular velocity + projected gravity` 全部回到支持域后再交权，5/5 仍按 `tilt → height collapse` 失败。
- 交权时 joint position 已 5/5 离开成功闭环支持，root 约 0.02 s 后 5/5 稳定离域；actual issued action 并没有先稳定 OOD。
- 因此问题不是单纯“交权太早”或“actor 首步动作坏”，而是 brake / transition 没有把完整 q/root/action-history 送入 recovery 学过的闭环流形。

### Phase26 dynamic bridge

- 物理 integration state bitwise exact、controller snapshot exact；两个 fork 的未来 10 tick exact。
- 唯一 CEM：1.0 s、5 个低维腰腿 mode×2 knots、16×3=48 candidates、seed2601。
- cost `4.912 → 3.254`（改善 33.8%）；root/gravity 与 2 s root safety 均改善。
- 但 bridge 末端 joint-position 距离 `1.955`、previous-action `1.050`（门为 `<=1`）。
- 交权后 1 s 的 gravity/q/previous-action/root 最大距离分别 `1.368/2.329/2.731/1.471`，联合支持域失败。

BASE 裁决：**机制信号存在，但 dynamic bridge teacher 不成立；不解锁 recovery PPO、25-update 或长训。**

## WBT / Any2Any 路线

### Phase36 后的真实前提

- 修正 Silver contact provenance 后，train GMR true Silver=`0`，held-out Silver/Gold=`0`。
- Stage30 历史 Silver 已降级为 Bronze，不能作为 optimizer 数据门。

### Phase37–40：几何修复的上限

- official contact 明显增加，并能消除全 sole 穿地。
- 但同一个 DS→右脚单支撑→DS 周期始终无法同时满足 `contact max <=0.5 mm` 与 `stance speed <=0.10 m/s`。
- Phase40 放开 root XY 后，stance speed 仅 `0.309→0.251 m/s`，contact max `1.035→0.851 mm`；仍无硬可行证书。

### Phase41：GRF / centroidal oracle

- 固定 Phase40 的 23 个有效导数帧：geometry-qualified `0/23`，force-LP feasible `0/23`，jointly feasible `0/23`。
- 即便有候选接触点的帧，也不能用非负法向力与 μ=1 摩擦锥满足六维合力/力矩平衡。
- COM acceleration p95/max=`6.657/8.469 m/s²`；centroidal momentum rate p95/max=`33.292/37.233 N·m`。

### Phase42：contact-first centroidal teacher

- 变量：25 帧 COM xyz + active sole GRF；硬约束包括离散线动量、线性化零净矩、非负法向、μ=1 摩擦锥、COP/足底与 COM 连续边界。
- 第一轮 DAQP 无解；完全相同的 `1371 variables / 144 equalities / 4926 inequalities` 用 HiGHS 独立可行性检查仍为 infeasible，排除单一 QP 后端故障。
- 初始零净矩残差 `171.4 N·m`，水平/垂直 COM acceleration `4.206/8.402 m/s²`。
- A 层失败，因此没有用 soft reward 绕过，也没有运行 whole-body IK、physics 或 PPO。

WBT 裁决：**当前 contact schedule、COM 边界与源动作语义的组合不能生成硬可行 teacher；Any2Any PPO 数据门为空。**

所有 COM、GRF、COP、contact 与 momentum 均为 official MuJoCo 模型估计，不是 X2 实机测量真值。

## 长训解锁矩阵

| 要求 | BASE | WBT |
|---|---|---|
| 冻结官方物理合同 | 通过（test-only bridge 边界另行标注） | 通过 |
| teacher / transition 本身可行 | 失败 | 失败 |
| train 数据门非空 | 有旧状态数据，但无可用 bridge teacher | true Silver=0 |
| 独立 held-out 资格 | 不适用/未达到 | 0 |
| official 严格门改善 | 无 stop/full 改善 | 无可跑 teacher |
| 允许 25-update / PPO | 否 | 否 |
| 允许长训 | **否** | **否** |

## 下一步不再做什么

- 不继续扫 LoRA、reward、contact tolerance、root-z、trust region、CEM population/horizon 或 handoff wait。
- 不把 Stage250 nominal 视频冒充 strict matched-event 后端。
- 不把 2 s 不倒、cost 下降或软接触改善冒充可用 teacher。
- 不在 train Silver=0 时启动 Any2Any PPO。

## 下一次重新开启实验所需的新信息

至少满足一项，才值得解除当前停止门：

1. X2 官方/原生稳定 locomotion 或全身控制 policy 及其真实 command contract；
2. 能同步给出 X2 接触/足底力或经官方模型验证的动态 teacher 数据；
3. 重新定义的目标机器人接触时序与foot placement，而不是继承当前 PHUMA/GMR schedule；
4. 成熟的全身直接配点工具链，允许同时优化 q/root/contact force/torque，并先在局部周期给出硬可行证书；
5. 对 BASE，新的 bridge 表示必须显式覆盖 joint-position 与 actual action-history 可达性，而不是只优化姿态/root。

## 证据入口

- `reports/baseline/x2_recovery_phase26_bridge_cem.md`
- `reports/retarget/x2_wbt_centroidal_force_feasibility_phase41.md`
- `reports/retarget/x2_wbt_contact_first_centroidal_teacher_phase42.md`
- `reports/X2_DUAL_TRACK_PHASE24_40_CHECKPOINT_20260809.md`
- `reports/baseline/x2_base_report_video_manifest.md`

## 状态

- 当前不是项目完成，也不是迁移成功。
- 当前是**长训前硬门失败并已正确停训**；保留现有 nominal BASE 能力与全部可追溯证据。
- 下一步先封版、Git 推送并执行第10任务节点百度大文件归档。
