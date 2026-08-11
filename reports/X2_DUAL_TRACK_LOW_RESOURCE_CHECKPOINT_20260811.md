# X2 双路线低资源检查点（2026-08-11）

## 项目状态

当前两条路线均继续，但都未解锁 RL/长训：

| 路线 | 已证实的改善 | 当前硬阻塞 | 下一唯一高信息量动作 |
|---|---|---|---|
| 动态重定向 | reset把fall `.575→.683s`；Phase15将足距p50 `.801→.464m`并修复DS覆盖 | 原SS仍137/137不成立；形态缩放需root-z改10.4cm且丢单支撑语义 | 以Phase15仅作初值，联合重做足位、contact、root/COM/q |
| BASE recovery | exact snapshot、stateful suffix、handoff continuity、低维 bridge 机制均已验证；Phase26 cost 降 33.8% | 没有进入任何带连续成功未来的联合物理盆地；Phase39复制真实suffix仍失败 | 只考虑 state-conditioned 足端/载荷/root 闭环 bridge；不再复读离线PD序列 |

## 本次 BASE Phase38 新结论

Phase19-v2 中存在 215 个“当前及未来 1 秒全部 success-safe”的候选起点。用 Phase25 的 robust LOO-p95 标定审计 Phase26 best endpoint 后：

- 到任意连续目标的最近 q 距离仍为 `1.955×` 门限；
- 到任意连续目标的最近 previous-action 距离仍为 `4.209×`；
- 等权联合最优来自 Phase19 r4 tick354，但 composite 仍为 `3.632×`，没有入域。

所以 Phase26 旧 cost 会把来自不同时间/episode 的最近邻拼在一起，形成物理上未必有连续未来的目标。下一步必须改变目标表示，不能只增加 population/iteration/seed。

Phase39 已进一步执行唯一一次无搜索 sequence-guided bridge：1 秒 bridge 自身安全，previous-action 回到 `0.842×` 门内，但 projected gravity 达到 `13.744×`；f005 接管后 0.34 秒 tilt 越界、0.58 秒高度坍塌。由此排除“把成功 episode 的 PD target/history 平滑复制过来就能进入同一盆地”。BASE 下一表示必须依据当前 state/contact 闭环调整，不能换另一个 suffix 重跑。

## 资源合同

- 默认单主任务、串行推进，不再开启多路线 subagent。
- 默认 CPU 单进程、低线程、0 GPU；仿真或训练前检查共享设备负载。
- 一次只运行一个预注册实验；失败先停并形成报告。
- 当前不启动 PPO、长训、AimDK ROS 批量 rollout。
- 本阶段只读审计实际使用：0 physics step、0 optimizer step。

## 版本与备份

- 上一次百度完整归档：`dynamic_retargeting/x2_dynamic_retargeting_race_full_20260811.tar.gz`
- SHA256：`219a1dc5739bfdd5078c656055f04df151d225202ee76ef6a2541ef788bde683`
- 归档后实质任务计数：`8/10`
  1. Phase10 actuator mapping correction；
  2. Phase11 name-mapped event SBTO；
  3. BASE Phase38 sequence-consistent target audit。
  4. BASE Phase39 single sequence-guided bridge。
  5. Dynamic Phase12 task-space/load pre-physics audit。
  6. Dynamic Phase13 full support-margin audit。
  7. Dynamic Phase14 contact-label-only repair audit。
  8. Dynamic Phase15 deterministic morphology normalization。
- 下一百度大包：完成第 10 个实质任务时触发；目前还剩 2 个。
- Git：重要合同纠正和小型审计即时提交，不等待第 10 项。

## Phase38 证据

- [BASE Phase38 报告](baseline/x2_recovery_phase38_sequence_target.md)
- [Phase38 JSON](official_x2/phase38_sequence_consistent_bridge_target.json)
- [审计工具](../tools/official_x2/audit_phase38_sequence_consistent_bridge_target.py)
- [回归测试](../tests/test_phase38_sequence_consistent_bridge_target.py)
- [BASE Phase39 报告](baseline/x2_recovery_phase39_sequence_guided_bridge.md)
- [Phase39 JSON](official_x2/phase39_sequence_guided_bridge.json)
- [Dynamic Phase12 报告](../research/dynamic_retargeting_20260811/phase12_taskspace_load_bridge/REPORT_phase12_taskspace_load_bridge.md)
- [Phase12 preflight](../research/dynamic_retargeting_20260811/phase12_taskspace_load_bridge/phase12_preflight.json)
- [Dynamic Phase13 报告](../research/dynamic_retargeting_20260811/phase13_support_margin_audit/REPORT_phase13_support_margin.md)
- [Phase13 result](../research/dynamic_retargeting_20260811/phase13_support_margin_audit/phase13_result.json)
- [Dynamic Phase14 报告](../research/dynamic_retargeting_20260811/phase14_contact_schedule_repair/REPORT_phase14_contact_schedule_repair.md)
- [Phase14 result](../research/dynamic_retargeting_20260811/phase14_contact_schedule_repair/phase14_result.json)
- [Dynamic Phase15 报告](../research/dynamic_retargeting_20260811/phase15_morphology_normalization/REPORT_phase15_morphology_normalization.md)
- [Phase15 result](../research/dynamic_retargeting_20260811/phase15_morphology_normalization/phase15_result.json)

当前总裁决：`BOTH_TRACKS_MECHANISM_POSITIVE / NO_TEACHER / NO_TRAINING_UNLOCK`。
