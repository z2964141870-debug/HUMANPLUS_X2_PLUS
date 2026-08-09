# X2 WBT Phase28：固定24条 official-v1 Bronze/Silver资格审计

## 裁决

- 完成：`24/24`；blocked：`0`。
- tier：Silver `0` / Bronze `2` / Reject `22`。
- train Silver：`0`；held Silver/Gold：`0`。
- WBT PPO 解锁：`False`。本阶段 optimizer/training/physics-step/真机均为 0。
- contact、COM、FK 均为模型估计，不是真实 GRF/COP/足底力。
- Bronze 失败集中在 tracked-keypoint p95 `20/24`、joint-step p95 `13/24`、severe self-collision `10/24`。
- 即使暂时忽略 Bronze 前置门，22 条动态动作中 stance speed 与 stance excursion 均为 `22/22` 失败，contact timing 为 `21/22` 失败；能独立通过全部 Silver 子门的动作仍为 `0`。

## 冻结离线指标合同

- threshold revision：`False`；逐 clip 调参：`False`。
- keypoint：整段一次 Umeyama similarity fit 后，计算 pelvis/torso/双脚/双腕残差。
- source contact intent：对齐后的 source foot z 不高于双脚 p05 ground + 20 mm。
- official contact：Phase7 active-sole signed distance 不高于 reset clearance + 1 个 sole-sphere radius。
- stance speed/excursion 使用 official ankle-roll FK；contact timing 是同脚 source event 到最近 target event 的 p95。
- severe self-collision：两个非 world body 的官方碰撞对 penetration >30 mm；仍只是模型碰撞证据。

## 漏斗

| ID | split | adapter | frames | old/official max | tier | reject reasons |
|---|---|---|---:|---:|---|---|
| `AMASS-STAND-001` | train_candidate | amass_smplx_v4 | 90 | 0.000e+00 | Bronze | `silver:static_exception_not_dynamic_silver` |
| `AMASS-UPPER-001` | train_candidate | amass_smplx_v4 | 172 | 0.000e+00 | Bronze | `silver:static_exception_not_dynamic_silver` |
| `AMASS-WALK-001` | train_candidate | amass_smplx_v4 | 228 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:sole_penetration_p95; bronze:tracked_keypoint_p95` |
| `AMASS-TURN-L-001` | train_candidate | amass_smplx_v4 | 280 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:joint_step_max; bronze:sole_penetration_p95; bronze:sole_penetration_max; bronze:tracked_keypoint_p95` |
| `AMASS-TURN-R-001` | held_out | amass_smplx_v4 | 249 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:sole_penetration_p95; bronze:tracked_keypoint_p95` |
| `AMASS-SQUAT-001` | train_candidate | amass_smplx_v4 | 165 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:joint_step_max; bronze:tracked_keypoint_p95` |
| `AMASS-KICK-L-001` | train_candidate | amass_smplx_v4 | 136 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:joint_step_max; bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max` |
| `AMASS-KICK-R-001` | held_out | amass_smplx_v4 | 86 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:joint_step_max; bronze:sole_penetration_p95; bronze:sole_penetration_max; bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max` |
| `AMASS-COORD-FAST-001` | held_out | amass_smplx_v4 | 289 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:joint_step_max; bronze:sole_penetration_p95; bronze:sole_penetration_max; bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max; bronze:no_severe_self_collision` |
| `AMASS-FAIL-CROUCH-001` | train_candidate | amass_smplx_v4 | 261 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:sole_penetration_max; bronze:tracked_keypoint_p95; bronze:no_severe_self_collision` |
| `AMASS-FAIL-CIRCLE-001` | held_out | amass_smplx_v4 | 495 | 0.000e+00 | Reject | `bronze:sole_penetration_p95; bronze:tracked_keypoint_p95; bronze:no_severe_self_collision` |
| `AMASS-FAIL-THROW-001` | held_out | amass_smplx_v4 | 294 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:joint_step_max; bronze:tracked_keypoint_p95` |
| `PHUMA-LUNGE-R-001` | train_candidate | phuma_g1_fk_full_hierarchy | 120 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:tracked_keypoint_p95` |
| `PHUMA-LUNGE-L-MIRROR-001` | train_candidate | phuma_g1_fk_full_hierarchy | 120 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:tracked_keypoint_p95` |
| `PHUMA-RAISE-L-001` | train_candidate | phuma_g1_fk_full_hierarchy | 73 | 0.000e+00 | Reject | `bronze:no_severe_self_collision` |
| `PHUMA-RAISE-R-001` | held_out | phuma_g1_fk_full_hierarchy | 67 | 0.000e+00 | Reject | `bronze:tracked_keypoint_p95; bronze:no_severe_self_collision` |
| `PHUMA-SQUAT-001` | train_candidate | phuma_g1_fk_full_hierarchy | 120 | 0.000e+00 | Reject | `bronze:semantic_review` |
| `PHUMA-FAIL-MOVE28-001` | held_out | phuma_g1_fk_full_hierarchy | 120 | 0.000e+00 | Reject | `bronze:tracked_keypoint_p95` |
| `PHUMA-COORD-001` | held_out | phuma_g1_fk_full_hierarchy | 112 | 0.000e+00 | Reject | `bronze:tracked_keypoint_p95` |
| `BONES-WALK-FAIL-001` | train_candidate | bones_soma_bvh_full_hierarchy_right_relative_yaw | 359 | 0.000e+00 | Reject | `bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max; bronze:no_severe_self_collision` |
| `BONES-TURN-FAIL-001` | held_out | bones_soma_bvh_full_hierarchy_right_relative_yaw | 292 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:joint_step_max; bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max; bronze:no_severe_self_collision` |
| `BONES-STOP-FAIL-001` | train_candidate | bones_soma_bvh_full_hierarchy_right_relative_yaw | 362 | 0.000e+00 | Reject | `bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max; bronze:no_severe_self_collision` |
| `BONES-SIDE-FAIL-001` | train_candidate | bones_soma_bvh_full_hierarchy_right_relative_yaw | 359 | 0.000e+00 | Reject | `bronze:joint_step_max; bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max; bronze:no_severe_self_collision` |
| `BONES-JOG-FAIL-001` | held_out | bones_soma_bvh_full_hierarchy_right_relative_yaw | 333 | 0.000e+00 | Reject | `bronze:joint_step_p95; bronze:tracked_keypoint_p95; bronze:tracked_keypoint_max; bronze:no_severe_self_collision` |

## 下一步

WBT PPO remains locked. Do not use the native Gold sanity seed to bypass missing GMR Silver; review the uniform generator/tier failures before selecting one new data intervention.
