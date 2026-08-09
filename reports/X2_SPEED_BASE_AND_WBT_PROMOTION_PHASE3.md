# X2 速度保底线与 WBT 晋升线 Phase3 裁决

日期：2026-08-09
状态：**BASE 获得相位一致事件的明确正信号但未过严格门；WBT 仅证明直行短窗 teacher 存在，仍无 Silver/Gold。**

## 一句话结论

BASE 将随机时钟控制的 `1/5` 提升到相位一致单更新的 `4/5`，证明完整事件语义是有效方向，但五更新退化到 `2/5`，长训继续锁定；WBT 的显式接触约束 teacher 只把单条直行生存从 `1.557` 提到 `2.456 s`，五动作仅 `1/5` 通过且仍会倒，不能生成训练数据或启动 Any2Any PPO。

## 主线 A：BASE

- 假设：物理 reset 与随机 command/event clock 错配，破坏起步—减速—停车闭环学习。
- 干预：固定 event time 0、完整 10.24 s rollout、在双支撑结束减速，并训练 terminal bilateral contact。
- 对照：同源、同 seed、同单更新的 random-clock 分支。
- 结果：control `1/5`；aligned `4/5`；aligned 5-update final `2/5`。
- 结论：相位一致有效，但继续 PPO 会以官方停车可靠性换取训练 reward/episode length；不进入 25-update。
- 下一步：从官方通过/失败停车 trace 构造 stateful RSI，原子恢复 physical state、event clock、gait phase 和 last action，只训练独立 stand/recovery 吸引域。

详见：[BASE Phase3](baseline/x2_transition_event_phase3.md)。

## 主线 B：WBT

- 假设：显式接触约束轨迹优化可以在官方 free-root 物理中产生短时 dynamic teacher。
- 干预：联合优化 COM/root、接触切换、stance terminal、swing clearance、landing terminal；先 walk smoke，再固定五动作面板。
- 对照：current-v4 原 reference，同一官方 AimDK v1.0 free/prescribed-root 物理。
- 结果：walk `1.557→2.456 s`、slip `0.00875→0.00663 m/s`；但 turn-right 仅 `+0.080 s`，stand→walk `-0.039 s`，walk→stand 虽 `+0.149 s` 但 slip 退化，turn-left 离线 0/12；总计 `1/5`。
- 结论：只得到单 clip、单 seed、确定性 physics 的 existence smoke，且 2.456 s 后仍倒；不能称 Silver/Gold。
- 方法学发现：reference frame0 qdot 与仿真强制零 qvel 存在显著错配；四类非直行动作的 model-contact/official-contact match 仅 `0.680–0.852`；当前 teacher 缺 root yaw、foot orientation、centroidal angular momentum。
- 下一步：先修初始 qvel、接触事件与转向姿态契约，再预注册同一五动作面板；不扩 CEM、不启动 PPO。

详见：[WBT Phase3](retarget/x2_contact_constrained_dynamic_teacher_phase3.md)。

## 晋升状态

- `BASE_LOCOMOTION`：冻结，不变。
- `BASE_TRANSITION`：冻结；Stage347 aligned-u1 仅保留为诊断候选，不晋升。
- Stage348 aligned-u5：否决，不保留为候选。
- stiff-fixed：尚未达到 5/5；stiff-fast 和完整矩阵不解锁。
- WBT Silver/Gold：0；忠实 Any2Any PPO 不解锁。
- 真机：未发送命令、未升级固件/SDK。

## Phase4 方法学更正（不改写 Phase3 原始结果）

后续审计确认，Phase3 的 `1/5→4/5` 使用的是同一个**旧部署门**，因此它仍能证明
aligned-u1 相对 random-clock control 的提升；但这个门并没有复现训练事件：旧门是
立即给 0.3 m/s、行走 4 s、再由 brake supervisor 停车，而训练是 1 s 加速、
4.2 s 巡航、2 s 减速。把官方评估改成真实 matched event 后，aligned-u1 和冻结
Stage306 分别为 0/5 与 0/5。故 Phase3 的 4/5 不能解释成“训练事件已经在官方域
通过”，长训锁定决定不变。完整证据见
[Phase4 matched-event audit](X2_MATCHED_EVENT_CONTRACT_PHASE4.md)。
