# BASE Phase39：真实成功 suffix 引导 bridge

## 裁决

`SEQUENCE_GUIDED_BRIDGE_REJECTED / TRAINING_LOCKED`

Phase39 按 Phase38 冻结的唯一目标运行一次，没有重试、CEM、optimizer、GPU 或参数扫描。它证明：把另一条成功 episode 的 physical PD target 与 action history 平滑接入，能够修正动作历史，却不能把当前 root/IMU/q 动力学状态带入同一个平衡盆地。

## 冻结合同

- 初态：Phase26 同一 official-MJCF integration/controller snapshot，SHA256 `c59c7c95d88e0080ee28bfd35bbd06c3b55e26acb3e24cef7c1088280e441391`。
- 目标：Phase38 选出的 Phase19 r4 tick354–404，共 51 个逐哈希 success-safe tick。
- 干预：1.0 s 内从 live brake physical target 以 C2 接入记录的 lower15 physical target；root 从未 prescribed/teleport。
- upper：保持原 fixed-upper live 路径。
- history：每 tick 按实际下发 physical target 反算并同步 previous/issued action。
- 交权：与 Phase26 相同，0.5 s blend 给同一 f005 recovery actor；报告 2 s、严格门看首 1 s。
- 成功门完全沿用 Phase26：末端所有组进入 success-safe、交权后 1 s 所有组留在 union support、root 全程安全。

## 结果

### Bridge 末端

`<=1` 才通过对应 success-safe 门。

| 组 | 归一化距离 | 判定 |
|---|---:|---|
| base linear velocity | 1.259 | 失败 |
| base angular velocity | 1.319 | 失败 |
| projected gravity | 13.744 | **严重失败** |
| joint position | 2.416 | 失败 |
| joint velocity | 0.180 | 通过 |
| previous action | 0.842 | **通过** |
| root posture | 1.040 | 轻微失败 |

bridge 自身 1 秒仍安全：root-z 最低 `0.6164 m`，tilt 最高 `0.1438 rad`。因此这不是“过渡过程中立刻摔倒”；失败在于末端虽同步了 action history，却已经有很大的 gravity/角速度/q 闭环不一致。

### f005 接管后

- tilt `>0.30 rad`：交权后 `0.34 s`；
- root-z `<0.55 m`：交权后 `0.58 s`；
- 2 s 最低 root-z：`0.1558 m`；
- 2 s 最大 tilt：`1.7110 rad`。

交权后首 1 秒所有关键组均明显离开 union support；joint position/joint velocity 的最大归一化离域甚至超过 `1000×`，属于倒地后的失效轨迹，不能当作 recovery 改善。

## 解释

Phase38 已说明旧五 mode bridge 没靠近任何连续成功 suffix。Phase39 进一步排除一个看似直接的修复：**单纯复制成功 suffix 的 PD target/history 也不够**。

原因是同一控制序列只在它原来的 state/contact/root 动力学上下文中成功。当前 snapshot 即使通过 C2 接到相同 lower target，root/IMU/contact 反应并不会自动变成 source episode；previous-action 进入门内而 gravity 达 `13.744×`，是这个结论的直接证据。

因此下一步若继续 BASE，不应：

- 换 r5 suffix 补跑；
- 扩 population/seed；
- 把 controller snapshot 直接 teleport 到不匹配的 physical state；
- 解锁 recovery 训练。

下一假设必须是 **state-conditioned dynamic bridge**：控制量需根据当前 root/IMU/contact 响应闭环调整，并把目标定义为可达的联合物理状态，而不是离线 PD 序列复读。是否继续实施，应先与动态重定向线的“直接足端/载荷转移表示”合并设计，避免两条线重复造同一种局部优化器。

## 资源与真实性

- 单进程、`nice 10`、BLAS/OMP 单线程、CPU；0 GPU。
- 唯一实际 candidate；失败后没有重试。
- direct unchanged official MJCF，同进程 test-only，不冒充 closed AimDK ROS。
- 0 policy training、0 PPO、0真机。

## 文件

- runner：`tools/official_x2/run_phase39_sequence_guided_bridge.py`
- result：`reports/official_x2/phase39_sequence_guided_bridge.json`
- target audit：`reports/baseline/x2_recovery_phase38_sequence_target.md`
