# Phase22 — X2 顺序接触骨架可行性

## 裁决

`CONTACT_SKELETON_COMPLETE / NO PHYSICS / NO TRAINING`

从 Phase41 的官方 AimDK 稳定状态出发，固定单一顺序

`DS_NATIVE → DS_LOAD → L_SUPPORT_R_SWING → DS_R_TOUCHDOWN → R_SUPPORT_L_SWING → DS_L_TOUCHDOWN → DS_SETTLE`

七个关键帧全部满足预注册硬约束。该结果证明官方 X2 模型在局部构型空间内存在左右交替离地、落脚和回到双支撑的几何接触骨架；它不证明时间连续、力可行、物理稳定、向前运动或动作语义。

## 固定合同

- 起点：Phase41 从 Stage250/Phase34 稳定闭环轨迹中确定性选出的 coherent native boundary。
- 每步变量：root 3D 平移、root 3D 局部旋转、lower12+waist3。
- 活跃支撑球：signed distance `0.25 mm`；全足底球不穿透。
- 摆脚：12 个 active sole sphere 全部至少 `12 mm`。
- COM：水平投影位于活跃足端质心中心。
- 已存在的支撑脚 XY 锚定；新落脚足仅在 touchdown 求解时自由，随后冻结。
- 单步 SLSQP 最多 100 次；无替代顺序、边界、阈值、初值或权重。
- 单进程、1 CPU thread、0 MuJoCo integration step、0 GPU、0 optimizer/PPO。

## 结果

- 完成关键帧：`7/7`。
- 最坏硬等式残差：`8.1514e-7`。
- 最坏硬不等式余量：`-6.4015e-11`，在预注册 `-1e-8` 数值容差内。
- `DS_LOAD` 的目标函数正常收敛；四个转换帧与 settle 达到硬可行，但 SLSQP 在 100 次上限停止。因此这些帧被标为 `objective_converged=false`，不能把它们写成最小改变量最优解。
- 起点到 settle 的 root-x 变化约 `20.28 mm`；左右 touchdown 足端 XY 基本回到原锚点。这是有意的 skeleton-only 结果，不是前进行走轨迹。

## 解释

Phase21 的单阶段问题把原生边界、接触、时间平滑和 PHUMA 语义同时压入一个非线性问题，最终严重失效。Phase22 将第一层缩为顺序接触骨架后，硬约束全部闭合，说明“接触构型本身不存在”不是当前主阻塞。

下一层应只加入一个新因素：为七个关键帧分配固定持续时间并生成连续轨迹，先检查速度、加速度和接触保持；通过后再加入前进步长/PHUMA 语义。不得直接把本结果送入物理或训练。

## 产物

- `phase22_sequential_contact_skeleton_contract.json`
- `run_phase22_sequential_contact_skeleton.py`
- `phase22_result.json`
- `tests/test_phase22_sequential_contact_skeleton.py`
