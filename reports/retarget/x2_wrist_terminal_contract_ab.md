# X2 Official Wrist Terminal Contract A/B

## 裁决

- candidate gate: **FAIL**。
- canonical wrist-roll position p95（左右最差）：`0.2560 → 0.2519 m`。
- joint-step p95：`0.1204 → 0.1202 rad/frame`。
- 腿腰 collateral RMS：`0.00521 rad`。
- 这里只验证运动学契约；没有训练，也没有把结果称为动态 Silver。

## 单变量

旧正收益 upper-hierarchy 的终端由 `wrist_yaw_link` 改为官方 WBT 观测契约的 `wrist_roll_link`；与该终端对应的 X2 elbow→wrist 几何长度同步更新。源动作、IK 权重、root、fps、平滑和求解器不变。

| variant | L wrist p95 m | R wrist p95 m | joint-step p95/max rad |
| --- | ---: | ---: | ---: |
| `wrist_yaw_terminal` | 0.21469 | 0.25605 | 0.12037/0.15706 |
| `wrist_roll_terminal` | 0.21286 | 0.25189 | 0.12021/0.15756 |

## 门禁

- wrist p95 至少改善 5%：`False`
- joint-step p95 不劣化超过 10%：`True`
- 腿腰差异 RMS ≤ 0.03 rad：`True`

大 cache 保存在 Git 外；本报告只记录方法、哈希和结果。
