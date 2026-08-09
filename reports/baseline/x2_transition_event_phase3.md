# X2 BASE Phase3：相位一致起步—行走—减速—停车事件

日期：2026-08-09
状态：**相位一致事件产生明确正信号，但 stiff-fixed 严格门仍未解锁；5-update 已证伪，不进入 25-update/长训。**

## 假设

旧训练允许随机 episode clock 和最多 7.2–8.0 s 的 command phase offset，但机器人物理状态始终从默认站姿 reset。这会让同一物理初态被赋予任意起步、巡航、减速或停车语义，使 actor 无法学习可复现的完整事件。

## 干预 / 对照

- 冻结源：Stage306 `model_2652.pt`。
- 共同事件：`stand=0, accelerate=1.0, cruise=4.2, decelerate=2.0, terminal hold=2.0 s`；事件在 `7.2 s` 的 gait double-support 结束；每 rollout `10.24 s`。
- 共同训练：seed 47、32 env、ideal actuator、stiff gain 1.2、fixed upper、Future-intent + response + transition adapter、base actor 冻结。
- control：随机 episode phase + 最多 7.2 s command phase offset。
- aligned：episode 从物理 reset 的 event time 0 开始、command offset 为 0，并在 terminal hold 训练双足接触目标。
- 物理裁决：官方 AimDK v1.0 MuJoCo、stiff-fixed、完整 stand→start→move→brake→stop，5 次重复。

## 结果

| 候选 | 官方完整门 | start | move | stop | 裁决 |
|---|---:|---:|---:|---:|---|
| 1-update random-clock control | 1/5 | 5/5 | 2/5 | 2/5 | 否决 |
| 1-update phase-aligned | **4/5** | 5/5 | 5/5 | 4/5 | 明确正信号，但未达 5/5 |
| 5-update phase-aligned final | 2/5 | 5/5 | 5/5 | 2/5 | 否决，不进入 25-update |

1-update aligned 的四次通过中，停车 drift 为 `0.070–0.145 m`、settle time 为 `2.32–2.70 s`；唯一失败在停车约 `2.1 s` 后倒地。5-update 的三次失败也全部发生在停车阶段，root-z 最低约 `0.139 m`，不是提前结束伪通过。

## 训练曲线与官方物理的反例

同一 5-update run 内：

- mean episode length：`134.77 → 442.58 step`；
- mean reward：`-0.13 → 17.55`；
- bad-orientation 日志：`0.699 → 0.470`；
- 官方完整门却从同一 run 的首 checkpoint `4/5` 降至 final `2/5`。

首 checkpoint 与独立 1-update run 的 SHA-256 完全一致：`4446ae8f...a8e4`，因此 4/5→2/5 是后续四次更新造成的真实变化，不是导出或随机初始化错位。

同时，训练日志中的 terminal double-support 项从 `-0.0053` 变为 `-0.0846`，contact-phase 从 `-0.0705` 变为 `-0.1106`。这些 episode 汇总会受存活时长影响，不能单独解释因果，但与官方停车退化方向一致。

## 方法学与基础设施自查

- 首轮门禁曾因容器缺少 `/repo/tools` 的 `PYTHONPATH` 而在适配器启动前失败；已修复并增加回归测试，所有裁决只使用修复后生成的有效 JSON。
- 官方 gate 的旧 70 s host timeout 会在 CPU 负载高时遗留容器；默认改为 180 s、加入 `--init` 和独立 `ROS_DOMAIN_BASE`。Stage347/348 未产生新遗留容器。
- Stage347/348 ONNX 的 PyTorch 对齐误差均约 `1.12e-7`；input `123D`、output `15D`，base actor 冻结。
- 未运行真机；训练 contact 是 IsaacLab 模型信号，不冒充真实足底力。

## 结论

相位一致不是边角修补：它把相同单更新、相同源 checkpoint 的官方成功率从 `1/5` 提高到 `4/5`，并让 start/move 达到 `5/5`。但继续到 5 updates 后停车退化到 `2/5`，说明当前 PPO 仍在用总体存活/速度收益换取 terminal contact/stand 吸引域。

因此：

- 保留 1-update aligned 作为**诊断候选**，不晋升为冻结 BASE；
- 5-update final 不保留为候选；
- 不运行 25-update 或长训；
- 下一次 BASE 实验必须针对停车边界状态恢复完整 event clock、gait phase 与 last-action 历史，训练 stand/recovery 吸引域；不再延长同一 actor。

## 证据

- `reports/official_x2/stage346_handoff_control_u1_gate.json`
- `reports/official_x2/stage347_handoff_aligned_u1_gate.json`
- `reports/official_x2/stage348_handoff_aligned_u5_gate.json`
- `scripts/run_stage345_transition_event_ab.sh`
- `src/cwi_x2/transition_schedule.py`
- `src/cwi_x2/transition_command.py`
