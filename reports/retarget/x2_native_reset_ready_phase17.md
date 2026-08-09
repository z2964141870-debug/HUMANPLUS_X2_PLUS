# X2 Reset-Ready Handshake Phase17

- 裁决：**PHASE17_READY_HANDSHAKE_CAPTURE_ABORTED_NO_PHYSICS**。
- ready 语义已从“订阅建立”修正为“完整31DOF state、odom、IMU均已实际到达”，静态合同和7项测试通过。
- 唯一一次 domain216 capture 在 ready 前超时；未发布 JOINT/RL、未生成 NPZ、未运行 prescribed/free physics。

## 假设

若 recorder 只在完整 actual state 可用后发 ready，再保持至少4秒 JOINT_DEFAULT，便能得到 Phase16 缺失的 reset-compatible、可评分前缀。

## 干预

- `ready-file` 不再在 ROS subscriptions 创建后立即写入。
- 只有 `ReplayBuffer.snapshot_ready()` 为真，即31个关节状态、odom、IMU齐备后才写 ready。
- 启动器固定 domain216，禁止 domain232；必须等 ready 才允许发布 JOINT_DEFAULT。

## 对照

Phase16 在 subscription-ready 后立即发 JOINT，虽然 mode receipts 相隔5.141秒，但首个完整snapshot过晚，实际可评分前缀只有0.882秒。

## 结果

| 时间点 | 结果 |
|---|---:|
| recorder ready | 未产生 |
| JOINT_DEFAULT mode | 未发布 |
| 首个 complete snapshot | 未形成/未落盘 |
| RL_DEFAULT mode | 未发布 |
| 可评分 JOINT prefix | 无资格 |

- launcher 的 ready 等待预算是 `100 × 0.05s = 5.0s`，随后报告 `recorder did not become ready` 并清理本次容器。
- Phase16 的首个完整snapshot约在 recorder elapsed `5.195s`，所以当前最可疑的是5秒边界过短；但单次结果也不能排除 zero/default mode 下完整state在JOINT前不可达。
- 无 domain216 残留进程；A3/domain232未触碰。

## 结论

- 结果：严格ready握手没有在5秒等待窗内发生，因此按照合同没有继续发布模式命令或运行physics。
- 结论：这是采集握手/等待预算失败，不是 reset-prefix、X2动力学或Any2Any的负结果。
- 下一步：本阶段不做第二次采集。若另行授权，唯一合理修复是延长ready等待窗，同时继续坚持“ready前绝不发JOINT”；保存资产明确证明完整前缀≥4秒后，才允许进入Phase15 prescribed gate。
