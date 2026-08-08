# 官方 X2 Stage306–315：动态意图、停车语义与长训前门禁

日期：2026-08-08
环境：AimDK v1.0 官方 X2 MuJoCo
候选：Stage306 transition-head，checkpoint s2652/s2657
结论级别：阶段甜点位；尚未满足完整长训解锁门

## 总体判断

当前已经把“站立→起步→直行→减速→停稳”在官方标称 PD 域做到 fixed/fast 上肢各 `3/3`，证明 X2 后端并非完全不可用；但 stiff 1.2× 域仍有概率性停车倒地，当前 s2652 的右转又存在系统性欠转，因此不能宣称跨条件稳定后端已经完成。

## Stage306：低维 transition head

**假设：** 原 actor 没有显式看到未来减速意图，单靠普通 response adapter 很难学会起步/停车事件。

**干预：** 冻结 Stage152/Stage298 主体，只训练 `6→16→8` 的 transition head；输入为 2 维 locomotion intent 与 4 维 gait phase，输出限制在腰腿 8 个协调模态内、幅值不超过 `±0.03`。

**对照：** Stage304 仅训练新增 observation columns；Stage306 使用独立 transition head。

**结果：** Stage304 动作变化仅 `mean=5.44e-5`、`max=0.001782`，属于无效支路；Stage306 的 s2647/s2652/s2657 首轮 stiff fixed/fast 全部通过，重复评估后：

| checkpoint | stiff fixed | stiff fast |
|---|---:|---:|
| s2647 | 1/3 | 3/3 |
| s2652 | 2/3 | 2/2（第 3 次为基础设施失败，未计） |
| s2657 | 3/5 | 4/5 |

**结论：** transition head 提高了通过概率，但首轮通过不能代表可靠；s2657 是本阶段概率较好的训练 checkpoint，仍未到稳定门。

**下一步：** 分析同 checkpoint 成功/失败停车轨迹，不继续盲选 checkpoint。

## Stage310：停车分叉分析

**假设：** 失败主要发生在移动策略向站立策略交权时。

**干预：** 对 s2652 stiff-fixed 的通过/失败 trace 做逐 tick 对照。

**对照：** Stage309 s2652 fixed-r2（通过）与 fixed-r3（停车倒地）。

**结果：** 两次都在 `0.76 s` 的第一个双支撑窗口交权；失败支路交权时仍带较大的横向/偏航动量。两次 rollout 从 prepare 初态已经有约 `5 mm` root-z 差异，随后在步态闭环中放大。

**结论：** “水平速度小＋模板双支撑”不足以证明状态进入站立策略的吸引域；官方实时 ROS 闭环也必须用重复率而非单次结果裁决。

**下一步：** 先做不训练的交权时序 A/B。

证据：[Stage310 分叉 JSON](stage310_s2652_fixed_stop_bifurcation.json)

## Stage311：延迟到下一个双支撑

**假设：** 多等待一个双支撑窗口能消除残余动量。

**干预：** 只把最早交权时间从 `0.5 s` 提高到 `0.9 s`，其余权重、控制器和门限冻结。

**对照：** Stage309 原交权逻辑。

**结果：** stiff fixed `1/5`、stiff fast `3/5`；多条 fixed 直到 `2.76 s` 才交权仍然倒地。

**结论：** 纯延迟交权是负结果；问题不只是“交得太早”。

**下一步：** 核对 transition head 的训练/部署意图语义。

## Stage312：只用训练同形的正向平滑减速

**假设：** 部署端复现训练的 `+0.3→0` C1 两秒速度日程即可稳定停车。

**干预：** 去掉反向速度制动，直接用正向 C1 减速后切换站立策略。

**对照：** 旧 `brake_blend_to_policy`。

**结果：** stiff fixed 前四次 `0/4`；除一次倒地外，其余虽站住但停车漂移达 `0.49–0.65 m`，远超 `0.15 m` 门限。

**结论：** 正向平滑减速缺少足够制动力，是干净负结果。

**下一步：** 保留底层反向制动，但把它与高层动态意图解耦。

## Stage313：控制命令与动态意图解耦

**假设：** 底层 actor 可以接收 `-K·v` 做反馈制动，但 transition head 应继续看到人体/任务层的“正向巡航→零”意图；不能把底层负反馈命令误当成高层动作意图。

**干预：** 新增独立的 current/future locomotion-intent 通道：

```text
底层 actor command: -K · measured_velocity
transition intent:   C1(+0.3 → 0), horizon=1.0 s
```

**对照：** Stage311 中二者共用同一个反向命令，导致 future delta 符号与训练相反。

**结果：** stiff fixed `3/5`、stiff fast `3/5`。相比 Stage312 的 `0/4` 明显恢复，但没有超过当前最佳 s2657 的总体重复率。

**结论：** 这是必须修正的控制契约错误，但不是 stiff 域稳定性的充分解。

**下一步：** 在 nominal 域确认基本闭环，再将该契约用于后续匹配长训。

## Stage314：官方 nominal 直行闭环

**假设：** 若主要剩余风险来自执行器增益边界，标称官方 PD 域应稳定通过。

**干预：** s2652、split-intent brake；PD multiplier `1.0`；fixed/fast 上肢各 3 次。

**对照：** 同模型 stiff 1.2× 的 Stage313。

**结果：**

| 条件 | 严格通过 | 停车最小 root-z | 停车漂移范围 |
|---|---:|---:|---:|
| nominal fixed | 3/3 | 0.616–0.621 m | 0.042–0.045 m |
| nominal fast | 3/3 | 0.616–0.620 m | 0.051–0.134 m |

**结论：** 官方 nominal 环境中的站立、起步、直行、减速、停稳已形成可复现闭环；当前失败不是“机器人完全不会动”，而是 stiff 执行器边界和航向能力仍不够稳。

**下一步：** 检查左右转向是否被 transition-head 训练破坏。

## Stage315：左右转向回归

**假设：** 当前 transition-head checkpoint 应保留旧后端左右转能力。

**干预：** nominal 域；右转 `wz=+0.15`，左转 `wz=-0.09`＋镜像；先跑 fixed，再预筛 fast。

**对照：** Stage252 旧 Stage219：fixed-right `3/3`、fixed-left `3/3`。

**结果：**

| 条件 | 严格通过 | yaw progress ratio |
|---|---:|---:|
| current fixed-right | 0/3 | 0.392–0.397 |
| current fixed-left | 3/3 | 0.724–0.764 |
| current fast-right | 1/3 | 0.400–0.520 |
| current fast-left | 1/1（仅预筛） | 0.755 |

所有已完成转向样本均未在停车阶段倒地；右转失败来自系统性欠转，而非生存或停车失败。

**结论：** 当前 checkpoint 不能替代 Stage219 的完整航向能力。transition-head 后续长训必须显式加入左右 yaw curriculum，并设置旧策略航向保持/蒸馏约束，避免只优化起停而遗忘右转。

**下一步：** 从 s2657 或 s2652 继续只训练小适配器，加入 start/cruise/decelerate/stop 与 `±yaw` 混合课程；按 +5/+10/+20/+30 更新筛选，候选必须同时通过 nominal/stiff、fixed/fast、左右转回归。

## 当前门禁状态

- [x] 官方 nominal：站立→起步→直行→减速→停稳，fixed `3/3`。
- [x] 官方 nominal：同闭环＋fast 上肢，`3/3`。
- [x] nominal 左转 fixed，`3/3`。
- [ ] nominal 右转 fixed：`0/3`，系统性欠转。
- [ ] stiff 1.2× fixed/fast：均只有 `3/5`。
- [ ] 继续 transition-head 长训并完成 checkpoint 曲线筛选。
- [ ] 四域及左右转完整回归。

因此，长训前准备已有实质推进，但完整解锁条件尚未达成。

## 当前基础设施阻塞

主机 NVIDIA 内核模块仍在，但 `/dev/nvidia*` 设备节点缺失，`nvidia-smi` 无法连接驱动；CPU 官方 MuJoCo 不受影响，IsaacLab GPU 长训暂时不能启动。系统自带的标准恢复工具为：

```bash
sudo /sbin/ub-device-create --verbose
nvidia-smi
```

该命令需要本机管理员密码。恢复后不需要重新采集数据，可直接执行下一支 transition-head 混合转向长训。

## 验证

- transition/future-intent 相关单元测试：`32 passed`。
- Stage314 官方 nominal 直行闭环：`6/6`。
- Stage315 已明确定位为右转语义退化，未通过完整门禁。
