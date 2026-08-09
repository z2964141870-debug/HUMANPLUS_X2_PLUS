# X2 WBT Phase36：Silver contact provenance更正

## 裁决

- **Phase30 `PHUMA-LUNGE-R-001` 的历史Silver无效，现更正为Bronze；train GMR Silver回到0，WBT optimizer/PPO资格撤销。**
- 历史Phase28/29/30报告和轨迹未覆写；本报告是追加式correction addendum。
- 0个physics integration step、0 optimizer；只对原cache做official x2.xml静态FK/collision复审。

## 根因

- Phase7 Bronze ground tolerance：`signed distance <= reset_clearance + radius = 0.01005m`。它用于容忍ground误差，不等于碰撞。
- Phase28/30 tier auditor却把该Bronze容差复用成Silver contact，因此把距地面0–10.05mm的悬空sole误标为contact。
- 更正后的Silver official contact：12 active sole spheres/foot与floor实际collision；与signed surface distance `<=0`逐帧完全一致。
- source foot-height标签只保留为intent，不能覆盖official geometry contact；两者是否一致仍沿用原transition与0.10s timing门，不新增/放松阈值。

## PHUMA-LUNGE-R-001

- historical→corrected tier：`Silver → Bronze`。
- source intent contact L/R：`{'left': 0.8685714285714285, 'right': 0.3485714285714286}`。
- historical tolerance-contact L/R：`{'left': 0.8285714285714286, 'right': 0.26857142857142857}`；因此旧报告得到flight=0。
- official collision-contact L/R：`{'left': 0.03428571428571429, 'right': 0.0}`；corrected flight=0.9657、transitions={'left': 5, 'right': 0}、DS→SS→DS=0。
- intent↔official frame agreement：`{'left': 0.1657142857142857, 'right': 0.6514285714285715}`；timing={'left': 2.0683333333333334, 'right': inf}。

## 重算漏斗

- Phase28固定24：`{'completed': 24, 'Silver': 0, 'Bronze': 2, 'Reject': 22, 'tier_changes': []}`。
- Phase29固定3：`{'completed': 3, 'Silver': 0, 'Bronze': 0, 'Reject': 3, 'tier_changes': []}`。
- Phase30固定3：`{'completed': 3, 'Silver': 0, 'Bronze': 1, 'Reject': 2, 'tier_changes': [{'id': 'PHUMA-LUNGE-R-001', 'historical': 'Silver', 'corrected': 'Bronze'}]}`。

| Phase30 motion | historical | corrected | corrected flight | official L/R contact | failed |
|---|---|---|---:|---|---|
| `AMASS-WALK-001` | Reject | Reject | 0.961 | `{'left': 0.02108433734939759, 'right': 0.018072289156626505}` | `bronze_prerequisite ; complete_cycle ; stance_speed_each ; clearance_p50_each_intended ; timing_each_intended ; intent_official_consistent_under_existing_event_gates ; flight` |
| `PHUMA-LUNGE-R-001` | Silver | Bronze | 0.966 | `{'left': 0.03428571428571429, 'right': 0.0}` | `complete_cycle ; transitions_each_intended ; stance_speed_each ; stance_excursion_each ; clearance_p50_each_intended ; timing_each_intended ; intent_official_consistent_under_existing_event_gates ; flight` |
| `AMASS-KICK-L-001` | Reject | Reject | 1.000 | `{'left': 0.0, 'right': 0.0}` | `bronze_prerequisite ; complete_cycle ; transitions_each_intended ; stance_speed_each ; stance_excursion_each ; clearance_p50_each_intended ; clearance_p95_each_intended ; timing_each_intended ; intent_official_consistent_under_existing_event_gates ; flight` |

## 结论 / 下一步

Phase30唯一train Silver来自contact审计provenance错误；官方geometry复审后降级，历史Silver与其后续optimizer资格正式撤销。该更正不否定1.46 retime对连续性/PD trackability的改善。

保持optimizer/PPO锁定；回到contact-feasible reference生成，不做root-z repair或阈值扫描。
