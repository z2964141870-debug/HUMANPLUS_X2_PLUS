# Stage 8.0：X2 最小原生后端门禁覆盖审计

日期：2026-08-07  
状态：`contract_frozen / baseline_recheck_pending`

## 结论

当前已有结果不能直接作为“起步—行走—转向—停止”的后端门禁。已有 Stage5/6/7
主要是固定速度连续运动窗口；完整 start/turn/stop 事件、统一接触/滑移、action
jerk 和停止稳定时间尚未形成同一评估矩阵。

因此当前覆盖为：**0/6 个核心事件具备完整 Stage8 证据**。这不是模型失败结论，
而是评估契约尚未补齐。

## 已有证据可复用部分

| 既有轮次 | 已覆盖 | 不能替代的部分 |
|---|---|---|
| Stage5 | 4 类上肢动作、0.20/0.30 m/s、nominal+delay、slow bounded upper | 没有起步/转向/停止；没有统一 foot slip、jerk、settle |
| Stage6 | BASE/FUTURE/FUTURE-NOPHASE、6 motions、0.20/0.30 m/s、nominal+delay | 仍是连续固定速度窗口；没有完整事件序列 |
| Stage7 | 三 seed 配对初态扰动、逐步共同窗口诊断 | 不是移动命令事件矩阵；不能证明转向和停止 |

## 当前冻结资产

- BASE：Stage208-s2550，SHA-256 已写入 `configs/x2_native_backend_gate_v1.json`；
- FUTURE/FUTURE-NOPHASE：仅作为已有候选对照，不作为正式基线；
- 上肢 slow 工作点：scale `0.25`、time scale `0.5`、幅值 `0.12 rad`、速度
  `0.20 rad/s`；
- 训练和真机开关均为 `false`。

## 下一次可执行实验

1. 固定 command/event 时间轴和执行器域哈希；
2. 在 fixed upper、nominal_delay 下先跑 BASE 的 stand/start/straight/turn/stop；
3. 记录缺失的 contact timing、stance slip、action jerk、stop settle；
4. 对同一初态/同一事件重放 FUTURE_NOPHASE 与 FUTURE_PHASE；
5. 再扩展 randomized 域、slow upper 和 fast upper 压力测试。

在第 2 步未完成前，不启动新的 PPO 或 AMASS 大规模重定向。
