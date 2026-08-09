# X2 WBT Contract Phase4

- 裁决：**CONTRACT_AUDIT_FOUND_REFERENCE_START_AND_ROOT_GROUND_MISMATCH**。
- 无训练、无 checkpoint、无真机；COM/DCM/contact 仍是官方 MuJoCo 模型估计。

## 1. frame0 qvel paired A/B

| role | zero(s) | ref-qvel(s) | Δ(s) | zero/ref slip | ref root lin/ang | ref joint p95/max |
|---|---:|---:|---:|---:|---:|---:|
| walk_straight | 1.557 | 0.928 | -0.629 | 0.0101/0.6290 | 0.332/25.221 | 2.239/2.680 |
| turn_left | 1.890 | 0.859 | -1.031 | 0.1490/0.1285 | 0.253/3.912 | 0.783/0.991 |
| turn_right | 1.117 | 0.593 | -0.524 | 0.0987/0.3546 | 0.265/5.019 | 0.900/1.193 |
| stand_to_walk | 1.603 | 0.747 | -0.856 | 0.0059/0.1282 | 0.100/2.680 | 0.347/0.385 |
| walk_to_stand | 1.952 | 0.735 | -1.217 | 0.0671/0.2321 | 0.649/7.012 | 0.922/1.208 |

## 2. official contact vs FK contact

| role | agreement | macro F1 | max event error(frames) | floor distance L/R p05(m) | collision L/R |
|---|---:|---:|---:|---:|---:|
| walk_straight | 0.219 | 0.005 | 24 | 0.0095/0.0021 | 0.000/0.050 |
| turn_left | 0.675 | 0.751 | 23 | -0.0582/-0.0704 | 0.896/0.883 |
| turn_right | 0.685 | 0.760 | 19 | -0.0536/-0.0595 | 0.900/0.967 |
| stand_to_walk | 0.788 | 0.852 | 22 | -0.0479/-0.0625 | 0.806/0.984 |
| walk_to_stand | 0.626 | 0.721 | 42 | -0.0514/-0.0623 | 0.888/0.976 |

## 3. turn orientation representation zero-update gate

| role | arrays exact | metric exact | survival(s) |
|---|---:|---:|---:|
| turn_left | True | True | 0.859 |
| turn_right | True | True | 0.593 |

## 结论边界

- 结果：reference-frame0 qvel paired mean survival delta=-0.851s; contact mismatch=True; turn zero-update exact=True.
- 结论：raw frame0 qvel 不安全；官方几何确认walk存在悬空、turn/start-stop存在厘米级穿地。当前contact事件首先反映root-z/ground生成器错误，不是X2动力学否证。
- 下一步：先修reference起始连续性与官方floor几何对齐，再重新提取official collision contact；turn非零panel继续锁定。

turn 表示的零更新门只证明新代码不破坏旧参考；它不证明非零 yaw/orientation teacher 有效。contact/FK 不一致属于生成器与标签契约问题，不能写成 X2 动力学不可执行。
