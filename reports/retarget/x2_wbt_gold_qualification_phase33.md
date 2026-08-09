# X2 WBT Phase33：Phase30唯一train Silver官方Gold资格门

## 裁决

- status：**PHASE33_PRESCRIBED_PASSED_FREE_REJECTED**；candidate prescribed：`True`；free executed：`True`。
- Phase28 original 与 Phase30 candidate 使用同一Phase12 runner/官方scene/control/gate；各自按同一规则从自身reference-exact frame0初始化，因此是paired qualification，不是纯因果单变量A/B。
- 不用physics调轨迹或1.46，不做CEM/PPO；prescribed失败只否定当前裸PD trackability。

## 结果

| reference/mode | survival | q RMSE | body pos/ori p95 | root pos/ori | contact | SS ref/real | slip p95 | sat/effort p95 | jump p95/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase28 original prescribed | 3.960/3.960s | 0.1099 | 0.1432m/0.5674rad | 0.00000m/0.00000rad | 0.934 | 0.000/0.131 | 0.411m/s | 0.0074/0.148 | 0.0441/0.1838 |
| Phase30 candidate prescribed | 5.800/5.800s | 0.0477 | 0.0813m/0.3080rad | 0.00000m/0.00000rad | 0.916 | 0.024/0.193 | 0.149m/s | 0.0000/0.065 | 0.0281/0.0671 |
| Phase30 candidate free | 0.575/5.800s | 0.1092 | 0.2209m/0.6247rad | 0.10148m/0.50392rad | 0.071 | 0.071/0.071 | 0.032m/s | 0.0000/0.363 | 0.0269/0.0597 |

- original prescribed gate：`False`，failed `['body_position_trackable', 'body_orientation_trackable']`。
- candidate prescribed gate：`True`，failed `[]`。
- candidate free gate：`{'checks': {'full_duration': False, 'q_trackable': True, 'root_xy_trackable': True, 'contact_agreement': False, 'slip_bounded': True, 'torque_saturation_bounded': True}, 'pass': False, 'failed': ['full_duration', 'contact_agreement']}`。

## 结论

候选可由官方裸PD跟踪，但free-root失败，证明静态Silver/trackability不等于Gold平衡；不否定Any2Any。

## 下一步

停止；保留为trackable Silver，不自动训练或调参。
