# X2 WBT Phase30：统一 1.46× time dilation

## 裁决

- Phase30 tier：Silver `1/3`、Bronze `0/3`、Reject `2/3`。
- candidate冻结：`True`；本阶段 physics/PPO/policy optimizer/真机均为 `0`。
- 即使产生 Silver，也只是静态 FK/contact Silver，不是官方 free-root Gold 或可部署策略。

## 1.46× 选择依据

只使用 train `PHUMA-LUNGE-R-001` 的 Phase29 门值，不使用 held-out。按时间缩放下界：
- qstep：`0.14117/0.10 = 1.4117`；root acceleration：`sqrt(5.969/4) = 1.2216`。
- contact timing 上界：`0.10/0.0667 = 1.5000`。
- 可行区间约 `[1.4117, 1.5000]`，中点 `1.4559`，预注册统一取 `1.46`。

## 结果

| motion | frames old→new | Phase29→30 tier | qstep p95 | root acc | timing L/R | stance speed L/R | rejects |
|---|---:|---|---:|---:|---:|---:|---|
| `AMASS-WALK-001` | 228→332 | Reject→Reject | 0.0824 | 2.626 | 0.133/0.675 | 0.459/0.249 | `bronze:tracked_keypoint_p95` |
| `PHUMA-LUNGE-R-001` | 120→175 | Reject→Silver | 0.0954 | 2.436 | 0.067/0.100 | 0.027/0.034 | `` |
| `AMASS-KICK-L-001` | 136→198 | Reject→Reject | 0.1063 | 2.466 | 0.540/0.507 | 0.038/0.081 | `bronze:joint_step_p95 ; bronze:tracked_keypoint_p95 ; bronze:tracked_keypoint_max` |

## 结论

统一1.46×时间扩展在 1/3 条 train 动作产生静态 Silver，单变量 existence test 通过；冻结该候选供独立评审，但它尚不是Gold、物理稳定或可部署证明。

## 下一步

Stop for review; do not automatically run physics or PPO.
