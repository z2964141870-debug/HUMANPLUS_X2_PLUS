# X2 WBT Phase31：冻结 repair＋retime held-out transfer

## 裁决

- held-out Silver：`0/3`；独立数据门：`False`。
- 三条动作在看指标前固定；没有替换动作、逐clip调参、空间重优化或time-factor扫描。
- Phase29权重/8轮/表示与Phase30 `1.46×` 完全冻结。
- 本阶段 physics/PPO/policy optimizer/真机均为 `0`；contact/FK不是实机GRF/COP/足底力。

## 结果

| motion | adapter | frames original→retime | original→repair→retime | qstep p95 | root acc | timing L/R | final rejects |
|---|---|---:|---|---:|---:|---:|---|
| `AMASS-TURN-R-001` | `amass_smplx_v4` | 249→363 | Reject→Reject→Reject | 0.0818 | 4.623 | 0.413/0.460 | `bronze:tracked_keypoint_p95` |
| `PHUMA-RAISE-R-001` | `phuma_g1_fk_full_hierarchy` | 67→97 | Reject→Reject→Reject | 0.0536 | 0.896 | 0.000/0.487 | `bronze:no_severe_self_collision` |
| `AMASS-KICK-R-001` | `amass_smplx_v4` | 86→125 | Reject→Reject→Reject | 0.1254 | 4.388 | 0.385/0.067 | `bronze:joint_step_p95 ; bronze:joint_step_max ; bronze:tracked_keypoint_p95` |

## 失败归因

- `AMASS-TURN-R-001`：1.46× 后 joint-step 已过门，但 keypoint p95 `0.1048 m` 仍略超 Bronze；更重要的是 counterfactual Silver 的 stance speed `0.380/0.222 m/s`、excursion `0.062/0.038 m`、timing `0.413/0.460 s`、flight `0.322` 和 root acceleration `4.62 m/s²` 同时失败，不是只差一个阈值。
- `PHUMA-RAISE-R-001`：支撑速度/漂移很好，但源动作在 original 就有 `0.881` severe-self-collision fraction，repair/retime 仍为 `0.881/0.876`；冻结的腰腿 repair 不具备修正上肢自碰撞的表示能力。右脚 timing 仍为 `0.487 s`，因此即便忽略自碰撞也不是 Silver。
- `AMASS-KICK-R-001`：repair 把 stance speed/excursion 压进门内，1.46× 也降低了 qstep/root acceleration，但最终 qstep p95/max `0.125/0.176 rad`、keypoint p95 `0.147 m`、左脚 timing `0.385 s`、flight `0.16` 与 root acceleration `4.39 m/s²` 仍失败。
- 这表明 Phase30 的 train-lunge Silver 没有独立迁移到转向、右抬腿或右踢腿；不能据此解锁 faithful Any2Any/PPO。

## 结论

冻结Phase29+30链在三条固定 held-out 上没有产生Silver；独立数据门仍锁定，不得以train Silver替代held-out证据。

## 下一步

Keep faithful Any2Any/PPO locked; do not substitute easier held-out motions.
