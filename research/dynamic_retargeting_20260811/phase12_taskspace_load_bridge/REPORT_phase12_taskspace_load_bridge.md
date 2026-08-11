# X2 Phase12：task-space / load-transfer bridge 物理前审计

日期：2026-08-11  
状态：**JACOBIAN AUTHORITY PASS / FULL LOAD TRANSFER IMPLAUSIBLE / STOPPED BEFORE PHYSICS**

## 目的

Phase8 的 joint-space SBTO 产生 26 次 contact chatter、最长离地 21 ms；Phase11 的平滑事件模式又退化成全程 stuck contact。Phase39 同时证明复制成功 episode 的 PD target/history 不能迁移平衡盆地。

本阶段检验两条路线共同的下一假设：从当前物理状态在线计算 task-space correction，固定右支撑足、抬左脚并把 COM 转移到右足，是否至少在局部运动学上可执行。若物理前已不成立，就不启动新的 rollout/search。

## 冻结合同

- official X2 scene SHA：`7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63`。
- Phase6 projected reset SHA：`141c6d7d9c006634eb00ddf25d2f593080abd0d8cfee8376767c14674d8cc2ea`。
- 控制变量严格 name-mapped lower12 + waist3，共15维；head不参与。
- 每脚只取12个 `contype!=0` official sole spheres。
- 任务：右足 centroid xyz、左足 centroid xyz、robot COM xy，共8维。
- 仅 `mj_forward` 一次；`mj_step=0`、optimizer=0、GPU=0。

## 结果一：自由度数量足够

| 任务 | Jacobian rank | lower15 nullity | condition |
|---|---:|---:|---:|
| 右支撑足 xyz | 3/3 | 12 | 6.00 |
| 双足 xyz | 6/6 | 9 | 17.20 |
| 双足 xyz + COM xy | 8/8 | 7 | 17.24 |

所以问题不是“X2少一个自由度”或 Jacobian 退化。当前姿态局部上能独立影响这些8个任务量。

## 结果二：所需载荷转移幅度不可执行

Phase6 reset：

- COM xy 与右足 centroid 的偏差为 `[0.0399, 0.4001] m`，范数 `0.4021 m`；
- 左/右足最小 clearance 为 `-0.500/-0.050 mm`，初态接触合同正常。

在线性化条件下要求“双足 centroid 完全不动，同时 COM 移到右足 centroid”，least-norm 解虽然任务残差仅 `1.3e-16 m`，但需要：

- lower15 delta RMS `1.298 rad`；
- 最大 delta `4.943 rad`（主要由 waist roll 吸收）；
- 2 个关节越限；
- 最大越限 `4.348 rad`。

这远超预注册最大 `0.35 rad`，物理前硬门失败。

## 结论

`FIXED-FEET LOAD-TRANSFER REPRESENTATION REJECTED BEFORE PHYSICS`

当前 lunge 的横向足位非常宽，而 reference 又要求右单支撑。保持双足位置不变再把 COM 推到右足，不是一个可执行的小修；这解释了：

- Phase8 只能产生短暂 chatter；
- Phase11 为保护右足与root，优化器选择不抬左脚；
- 单纯增加关节模式、population 或 feedback gain 不会解决几何幅度冲突。

下一动态重定向表示必须允许**接触时序和足位共同变化**：例如先移动/收回摆脚、允许受限 root/foot placement 重规划，或重新选择更可行的支撑转换；不能把原 reference 的超宽足位与右单支撑同时当硬目标。

本阶段没有运行物理，不能宣称新的 controller 成功或失败；它只诚实否定“固定双足足位的局部 task-space load bridge”。RL/teacher 继续锁定。

## 产物

- `audit_phase12_taskspace_load_bridge.py`
- `prereg_phase12_taskspace_load_bridge.json`
- `phase12_preflight.json`
- 本报告
