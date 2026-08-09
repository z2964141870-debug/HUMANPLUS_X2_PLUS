# X2 Activation Prefix Phase20

- 裁决：**PHASE20_ENVIRONMENT_PORT_COLLISION_NO_PHYSICS**。
- 静态协议通过，但唯一 domain219 capture 被共享端口冲突中止；未生成NPZ、未运行 prescribed/free physics。
- 这是环境并发冲突，不是对激活顺序、X2动力学、reference或Any2Any的实验结果。

## 假设

recorder订阅进程ready后先发JOINT激活telemetry；等首个真实complete snapshot，再额外保持JOINT至少4秒，才切RL，可以建立reset-compatible可评分前缀。

## 干预 / 对照

- 唯一协议变量：zero-mode完整ready前置 → subscription-ready后JOINT激活、complete snapshot后计时。
- 未改模型、PD、reference、门槛或事件replay方法。
- 静态顺序由测试锁定：subscription-ready → JOINT ack → complete-snapshot硬门 → 4秒 → RL ack。

## 结果

| 时间点 | 结果 |
|---|---:|
| subscription-ready | 达成，精确elapsed因无NPZ而不可恢复 |
| JOINT controller ack | 达成，ROS time `1786262192.047958s` |
| first complete snapshot | 未达 |
| complete后4秒prefix | 未达 |
| RL ack | 未发布 |

- 新官方sim在启动时明确报错：`0.0.0.0:51822 is already in use`，随后abort。
- 同时只读观察到另一agent容器 `x2-phase11_candidate_recovery_matched_stiff1p2_fixed_r1` 正在运行；未停止、未修改该容器。
- 因sim未存活，readiness在16.592秒仍为 state `0/31`、odom `False`、IMU `False`。

## Gate

- `simulator_healthy`: `False`
- `first_complete_snapshot_exists`: `False`
- `scoreable_joint_prefix_at_least_4s`: `False`
- `rl_segment_at_least_20s`: `False`
- prescribed/free executed：`False/False`

## 结论

- 结果：subscription-ready和JOINT ack虽已发生，但官方sim因51822端口冲突没有启动，无法产生telemetry或合格资产。
- 结论：激活协议仍未被测试；本次只证明多agent并发下官方sim的固定HTTP端口需要互斥。
- 下一步：本阶段不杀其他agent容器、不重采。若另行授权，先确认51822空闲，再原样运行同一协议，不引入新变量。
