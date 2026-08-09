# X2 双路线阶段检查点：BASE Phase24–25 / WBT Phase37–40

日期：2026-08-09

## 总体裁决

本轮把两条路线都推进到了一个可信的停止点，但尚未解决 X2 全身遥操作：

- **BASE 速度型后端**：更严格的物理状态交权门能够延迟倒地，却不能改变 `tilt → height collapse` 的失败闭环。交权瞬间的关节构型与 root 已脱离成功闭环支持域，因此不再继续扫交权时间或训练现有 recovery。
- **WBT / Any2Any reference 路线**：修正了 Silver 接触定义后，局部 reference repair 能显著增加官方真实接触并消除穿地，但仍无法同时满足支撑脚低速与接触等式。当前局部 FK / Sequential-QP 结构到达收益上限，train GMR Silver 仍为 0，PPO 保持锁定。

这不是“X2 不可能迁移”的证据；它说明下一步必须从局部几何修补升级为带接触力、质心状态和闭环动态的 teacher / trajectory optimization。

## BASE：交权门为何没有解决问题

### Phase24：完整物理支持域交权

- 冻结 Phase23 的模型、PD、brake、0.5 s blend 与动作历史支持门。
- 新增 `base angular velocity + projected gravity` 支持域条件，阈值来自 Phase19-v2 eligible 数据的稳健 LOO 分布，不按 episode 调整。
- 5/5 candidate 都满足完整门并实际交权；等待时间 0.36–1.58 s，中位 1.24 s。
- stand/startup 为 5/5，move/stop/full 仍为 0/5。
- root-z 坍塌绝对时刻中位数由约 4.74 s 延至 5.52 s，但交权后存活由 2.36 s 降至 2.28 s。

裁决：更晚倒地来自更久的 brake，不是 recovery 改善；“动作历史 + generator DS + 角速度 + gravity”仍不是安全交权的充分条件。

### Phase25：角色分布与最早偏离

- success-safe / critical-failure 冻结支持集对 gravity、previous action、issued action、joint position 与 root 有较强区分能力。
- Phase24 交权瞬间，joint position 已 5/5 超出 union 支持；root 在约 0.02 s 后 5/5 稳定离域。
- previous action、gravity 分别约 0.14 s、0.24 s 后才离域。
- actual issued action 在交权后 1 s 内没有稳定 OOD。

裁决：现有证据不支持“recovery actor 先输出异常动作把机器人带倒”；更符合证据的是 brake / transition 已把 q/root 送到 recovery 未覆盖的闭环状态。

## WBT：真实 official contact 修复到了哪里

### Phase37：全轨迹软约束修复

- 固定 PHUMA-LUNGE-R Phase30 的 175 帧、1.46× 时间、root XY/orientation、上肢、头部与接触意图。
- 只优化 root-z 与腰腿 15 DoF；Silver contact 使用 official x2.xml 的 12 sole spheres/foot，`signed distance <= 0`，不再使用错误的 10.05 mm Bronze 容差。
- official contact L/R 从 0.034/0 提升到 0.263/0.097，并首次形成一个 DS→SS→DS 周期。
- 但 flight 仍为 65.14%，stance distance、穿透、stance speed 与 swing clearance 未全部通过，最终仍为 Bronze。

### Phase38–40：硬约束局部证书

固定同一个 25 帧 DS→右脚单支撑→DS 周期，不更换窗口或门限：

| 阶段 | 表示 / 求解 | stance distance max | stance speed | 其他结果 | 裁决 |
|---|---|---:|---:|---|---|
| Phase38 | SLSQP，root XY 固定 | 1.272 mm | 0.144 m/s | 最大约束违反 1.465 mm | 无证书 |
| Phase39 | Sequential-QP/DAQP，root XY 固定 | 1.035 mm | 0.309 m/s | 全部 sole sphere nonpenetration 通过 | 无证书 |
| Phase40 | 同 Phase39，仅释放 root XY | 0.851 mm | 0.251 m/s | root XY 仅用 6.32 mm，rootacc/qstep/excursion 通过 | 无证书 |

原门为 contact max 0.5 mm、stance speed 0.10 m/s。释放 root XY 有改善，但不能单独解决支撑脚滑动；继续调 trust region、权重或 contact tolerance 的信息增益已经很低。

## 当前锁定状态

- BASE 25-update / 长训：**锁定**。
- WBT / Any2Any PPO：**锁定**。
- train GMR true Silver：**0**。
- held-out GMR Silver/Gold：**0**。
- 真机部署：**未解锁**。
- official AimDK MuJoCo：继续作为主要 held-out 物理域；所有 COM/contact 量均为模型估计，不冒充实机 GRF/COP。

## 下一条高信息增益工作

### 1. WBT：动力学感知 reference teacher

不再做局部 FK 权重扫描。对同一 25 帧周期建立带以下变量与硬约束的直接配点 / centroidal trajectory optimization：

- root SE(2)/高度与腰腿轨迹；
- 左右脚接触力、摩擦锥与接触互补；
- COM、centroidal momentum 与动力学一致性；
- 支撑脚零速、摆脚 clearance、q/dq/torque/连续性；
- official x2.xml 几何与执行器边界。

先只求一条局部可行证书；若 teacher 自身不可行，不进入 policy 学习。

### 2. BASE：brake→recovery 动态桥接 teacher

不再增加单帧 gate。冻结 recovery，使用短时 CEM/MPPI 或轨迹优化搜索一个 0.5–1.5 s 的 transition，使系统从 stop/brake 状态进入 success-safe 的 q/root/IMU/action-history 联合支持域，再交权。必须在 official MuJoCo 中检查：

- bridge 期间不倒、不滑；
- 交权后至少 1–2 s 保持在 success-safe 支持域；
- 不能以“不交权”或倒地后 drift 变小作为成功。

## 证据入口

- `reports/baseline/x2_recovery_phase24_physical_handoff.md`
- `reports/baseline/x2_recovery_phase25_role_aware_divergence.md`
- `reports/retarget/x2_wbt_official_contact_repair_phase37.md`
- `reports/retarget/x2_wbt_hard_constraint_certificate_phase38.md`
- `reports/retarget/x2_wbt_sequential_qp_certificate_phase39.md`
- `reports/retarget/x2_wbt_rootxy_sequential_qp_phase40.md`

## 版本管理

上一备份节点已完成：Git `42adf8f`；百度归档 `x2_phase36_dual_track_20260809.tar.gz` 上传命令成功，远端复查按用户约定留到每日手工检查。本检查点距离下一次“10 个实质任务”大文件归档尚不足，不重复上传。
