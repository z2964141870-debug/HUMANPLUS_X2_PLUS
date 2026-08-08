# X2 官方 MuJoCo Stage316–328：转向、Future-intent 与 stiff 域审计

日期：2026-08-08
模型：`stage306_s2652_transition_head_actor.onnx`
仿真：AimDK v1.0 官方 X2 MuJoCo
范围：只做部署契约与闭环门禁，不上真机，不把失败样本伪装成通过。

## 总体判断

本轮取得了一个真实但有限的甜点位：在 nominal PD 域，`0.5 s` Future-intent 停车预告保住直行 `6/6`，并把固定/快速上肢左右转面板做到 `7/8`；但 stiff `1.2x` 域的完全静止上肢分支仍只有 `3/5`，快速左转也仍有概率性欠转。因此长训前闭环比本轮开始时更完整，但还不能宣称跨条件后端已完成。

## 实验 1：右转增益标定（Stage316）

**假设：** 右转失败主要是官方部署侧 yaw authority 标定不足。
**干预：** 固定同一 checkpoint，仅扫描 `yaw_action_gain=2.5/3.0/3.5`。
**对照：** 同一 nominal 域、固定上肢、同一右转命令、每档 3 次。
**结果：** `2.5=3/3`，`3.0=3/3`，`3.5=0/3`；3.5 三次都完成转向但停车倒地。
**结论：** 右转欠转可由部署增益修正，安全甜点为 3.0；继续增大只会把航向问题转化成停车动量问题。
**下一步：** 左右方向采用独立增益，右 3.0、左 2.5。

## 实验 2：非对称左右转基线（Stage317）

**假设：** 对称单增益掩盖了 X2 左右转 authority 的结构差异。
**干预：** 右 3.0、左 2.5。
**对照：** fixed/fast 上肢 × left/right × 3 次。
**结果：** fixed `6/6`；fast `3/6`。失败包含欠转，以及一次左转完成后停车倒地。
**结论：** 固定上肢左右转已打通；剩余冲突集中在快速上肢扰动与转向/停车的耦合。
**下一步：** 核查 123-D 模型的 Future-intent 是否在部署中真的提前送入。

## 实验 3：Future-intent 停车预告（Stage320–321）

**假设：** Stage306 训练了未来速度变化，但正常 move 分支一直到 stop 阶段才送出减速意图。
**干预：** 在 move 末段向 123-D actor 显式提供平滑 future-vx；对比 `1.0 s` 与 `0.5 s`。默认值保持 0，防止静默改变旧模型契约。
**对照：** 与 Stage317 相同的 nominal 左右转面板，每组 2 次。
**结果：** `1.0 s=6/8`；`0.5 s=7/8`。Stage321 fixed `4/4`、fast-right `2/2`、fast-left `1/2`；唯一失败为 yaw ratio `0.461` 的欠转，停车仍通过。
**结论：** 1 秒预告过强；0.5 秒是净收益候选，它减少了快速转向的停车失稳，但没有消除快速左转的随机欠转。
**下一步：** 必须回归直行，防止为转向修复破坏基本移动闭环。

## 实验 4：nominal 直行回归（Stage325）

**假设：** 0.5 秒预告可能只在转向面板看起来更好，却破坏直行起停。
**干预：** 保持 Stage321 预告，只撤去转向命令。
**对照：** fixed/fast 上肢各 3 次，完整站立→起步→直行→制动→停稳。
**结果：** fixed `3/3`，fast `3/3`，合计 `6/6`；所有 startup/move/stop 子门均通过。
**结论：** 0.5 秒预告没有破坏 nominal 基础能力，可晋级为 nominal 部署甜点配置。
**下一步：** 进入 stiff `1.2x` PD 域重复评估。

## 实验 5：快速左转局部补丁反证（Stage322–324）

**假设：** 更高左 yaw 增益、停车保持上肢或末段 fade 可同时解决欠转与停车。
**干预：** gain 2.7/2.9；`upper_stop_mode=hold_last`；末段 `0.5 s` 撤去 turn feedback。
**对照：** 同一 fast-left、0.5 秒预告、每支 3–4 次。
**结果：** gain 2.7 与 2.9 均 `2/3`；hold-last `2/4`；gain2.9+fade0.5 `2/4`。高增益能消除欠转，但会概率性停车倒地。
**结论：** 这不是一个还能靠单参数扫开的宽裕区间；局部补丁只在“欠转”和“摔倒”间搬运失败。
**下一步：** 锁定保守左增益 2.5，不再继续无信息增益的参数扫描。

## 实验 6：stiff 1.2x 跨域回归（Stage326）

**假设：** 0.5 秒预告也能提高执行器偏硬域的停车稳定性。
**干预：** Stage321 预告配置进入 `Kp/Kd=1.2x`。
**对照：** fixed/fast 上肢各 5 次；旧 Stage313 两组均为 `3/5`。
**结果：** fixed `3/5`，fast `5/5`。fixed 两次失败均为停车倒地；fast 五次全部通过。
**结论：** Future-intent 明显改善 stiff＋上肢运动组合，但不能修复零上肢意图分支；执行器响应仍是独立硬缺口。
**下一步：** 拆分上肢 intent 条件与真实手臂物理运动，定位 fast 5/5 的来源。

## 实验 7：上肢 intent / 物理解耦（Stage327）

**假设：** stiff-fast 的收益来自 actor 读到未来上肢意图，而非手臂惯性。
**干预：** `intent-only`（模型看 0.25 intent、手臂不动）与 `physical-only`（手臂按 0.25 动、intent 置零）。
**对照：** stiff 1.2x，每组 3 次。
**结果：** intent-only `2/3`，唯一失败是 heading `0.323 rad` 略超门且未倒；physical-only `3/3`。
**结论：** fast 组稳定性主要来自真实手臂运动提供的全身平衡作用，intent 条件不是充分解释。
**下一步：** 测试更小、有界的平衡摆臂 residual 是否兼顾遥操保真。

## 实验 8：最小平衡摆臂幅值（Stage328）

**假设：** 很小的自主摆臂即可修复 stiff-fixed。
**干预：** intent 置零，仅施加 `upper_scale=0.10/0.15`；实际最大关节偏移分别约 `0.041/0.062 rad`。
**对照：** stiff 1.2x，每档 5 次。
**结果：** 0.10=`3/5`；0.15=`3/5`。两档都未超过无摆臂 fixed 的 3/5。
**结论：** 三次小样本的 3/3 是假甜点；最小摆臂 residual 未通过五次复验，不能接入默认系统。完整 0.25 上肢轨迹可稳定，但会改变“上肢静止”的控制契约。
**下一步：** 不再扫幅值；训练时应显式覆盖 stiff＋zero-upper 的 start/cruise/decelerate/stop，而不是依靠手臂动作碰巧稳定。

## 当前可锁定配置

- checkpoint：Stage306 `s2652`；
- nominal：Future-stop preview `0.5 s`；
- straight：lateral recovery supervisor，nominal fixed/fast `6/6`；
- turn：右 gain `3.0`，左 gain `2.5`；
- nominal turn：Stage321 `7/8`；
- stiff fast-upper：Stage326 `5/5`；
- 所有实验继续使用官方 AimDK v1.0 X2 MuJoCo 和现有严格 startup/move/stop 门禁。

## 仍未解锁长训的硬门

- stiff 1.2x、zero-upper：`3/5`，存在停车倒地；
- nominal fast-left：仍有概率性欠转；
- 尚未完成 soft/nominal/stiff × fixed/slow/fast × straight/left/right 的完整矩阵；
- 当前 NVIDIA 驱动通信失败，IsaacLab GPU 续训未启动。

## 下一轮训练的最小目标

从 Stage306-s2652 出发，只训练小适配器并保留基座保护约束；课程必须混合：

1. start / cruise / 0.5 s future-stop preview / decelerate / stop；
2. straight / left / right，左右分别保留已验证的部署增益；
3. nominal 与 stiff 1.2x，重点提高 stiff＋zero-upper 权重；
4. upper fixed / slow / fast，禁止只凭 fast 分支晋级；
5. 候选先过 nominal 直行 `6/6`、nominal 转向面板，再过 stiff fixed/fast 各至少 `5/5`，最后扩展完整矩阵。

本轮不应被解释成“X2 已经完全会走会转”，而应解释为：标称域的起步—行走—左右转—停车闭环已基本形成，原先模糊的动力学冲突被收窄为两个可训练目标——`stiff＋zero-upper` 停车吸引域和 fast-left 航向裕度。
