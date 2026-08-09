# X2 WBT Phase32：train-only gate-constrained full-WBT29 feasibility

## 裁决

- train Silver：`0/3`；未来重新评 held：`False`（硬门要求2/3）。
- full WBT29＋root参与；head2锁定。硬投影覆盖joint limits/step、root acceleration和rootXY，FK约束用逐轮激活线性化，最终仍由原Bronze/Silver门裁决。
- 没有soft reward扫权、逐clip调参、删段、fixed root、forced double contact、physics或PPO。

## 结果

| motion | Phase29→Phase32 | qstep p95/max | root acc | speed L/R | timing L/R | rootXY max | rejects |
|---|---|---:|---:|---:|---:|---:|---|
| `AMASS-WALK-001` | Reject→Bronze | 0.0950/0.0950 | 3.800 | 0.862/0.551 | 0.013/0.000 | 0.059 | `silver:stance_speed_each ; silver:stance_excursion_each` |
| `PHUMA-LUNGE-R-001` | Reject→Bronze | 0.0950/0.0950 | 3.800 | 0.334/0.824 | 0.000/0.028 | 0.064 | `silver:stance_speed_each ; silver:stance_excursion_each` |
| `AMASS-KICK-L-001` | Reject→Reject | 0.0950/0.0950 | 3.800 | 1.489/2.498 | inf/0.133 | 0.124 | `bronze:sole_penetration_p95 ; bronze:sole_penetration_max ; bronze:tracked_keypoint_p95 ; bronze:tracked_keypoint_max` |

## 硬门收敛与失败归因

- joint-step、joint limits、root acceleration、rootXY、head lock 的投影约束数值成立：最终三条 joint-step max 约 `0.095 rad`，root acceleration max 约 `3.80 m/s²`，rootXY修正仅 `0.059/0.064/0.124 m`。
- 非线性 FK 接触约束没有收敛。Walk 最后一轮仍有 keypoint/stanceXY/stanceZ/swing active constraints `247/244/93/31`，最终 stance speed `0.862/0.551 m/s`；Lunge 为 `40/136/10/2`，stance speed `0.334/0.824 m/s`。二者虽到 Bronze，但都因 stance speed，且各有一侧 excursion 超门，未到 Silver。
- Kick 的 active constraints 从 `773` 增到 `1008`，说明当前顺序线性化表示在大 keypoint/contact 冲突下发散；最终还出现 penetration、keypoint、flight `0.632` 和无有效左 stance 窗。它不是可行轨迹。
- 因此“约束被写入系统”不等于“硬约束已经兑现”：本轮只证明直接投影的时序/边界门可控，同时否定当前12轮顺序线性化接触可行性求解器。按任务门，不再退回soft-weight扫描。

## 执行完整性

- 第一次单进程在 Walk 第9轮因每轮重复完整SMPL审计导致内存观测开销终止；第二次移除该观测后在总墙时边界于 Kick 第7轮终止。两次均无artifact进入结果，也未据中间tier调参。
- 最终结果由三条从第1轮重新开始、各自完整12轮的独立worker shard组成，再只读汇总；配置、顺序和门禁一致。

## 结论

统一硬门full-WBT29求解在 0/3 条train产生Silver；未达到2/3，按门停止且held仍锁定。

## 下一步

Stop this configuration. Do not retune constraints or read held-out.
