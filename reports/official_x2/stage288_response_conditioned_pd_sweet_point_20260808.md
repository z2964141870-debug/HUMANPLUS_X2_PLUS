# Stage288：X2 执行器响应分支与 hard-cell 裁决

日期：2026-08-08

## 假设

Stage281 在官方 X2 MuJoCo 的 nominal PD 条件表现稳定，但 Stage284 在 stiff `Kp/Kd=1.2x` 下固定上肢仅 `1/3`、快速上肢 `0/3`。原因可能不是 Future-intent 信息不足，而是现有 adapter 在上肢意图为零时严格关闭，无法根据目标机器人实际 `q/dq/action history` 适配执行器响应。

## 干预

在冻结 Stage281 actor 和 Future-intent adapter 的前提下增加零初始化响应分支：

- 输入：部署可得的 15 维腰腿 `q`、`dq`、上一动作及 gait phase；
- 输出：仅通过既有低维髋腰协调基；
- 单分支幅值上限 `0.05`，合成 residual 上限仍为 `0.10`；
- Stage285 初版允许 pitch/roll/yaw 八个模式；
- Stage286 根据 trace 消融，响应分支禁止修改 hip yaw 与 waist yaw，仅保留 hip pitch/roll 和 waist roll；
- 没有修改官方门槛。

训练从 Stage281-s2623 开始，PD gain 随机化 `0.9–1.2`，快速上肢扰动，先跑 5 update 并逐 checkpoint 早停筛选。

## 对照

- Stage284 / Stage281-s2623：无响应分支；
- Stage285-s2624：响应分支包含 yaw；
- Stage286-s2624：同一 checkpoint，训练-free 禁止响应 yaw，用于结构消融；
- Stage287-s2625：从 Stage285-s2624 以 no-yaw 结构匹配微调 1 update；
- Stage288：Stage286 加入此前低速验证过的事件式 yaw recovery。

## 结果

| 模型/条件 | 行走 heading max | 行走门 | 停车门 | full gate |
|---|---:|---:|---:|---:|
| Stage285-s2624, stiff/fixed r1 | 0.282 | pass | pass | pass |
| Stage285-s2624, stiff/fixed 3 次 | 0.281/0.281/0.307 | 2/3 | 3/3 | 2/3 |
| Stage285-s2624, stiff/fast | 0.433 | fail | pass | fail |
| Stage286 no-yaw, stiff/fixed | 0.286 | pass | pass | pass |
| Stage286 no-yaw, stiff/fast | 0.328 | fail | pass | fail |
| Stage287 matched 1 update, stiff/fast | 0.290 | pass | **fall** | fail |
| Stage288 yaw recovery, stiff/fast | 0.307 | fail | **fall** | fail |

Stage285 早停扫描中只有 s2624 首次通过 stiff/fixed；继续到 s2625–s2627 分别出现轻微 heading 超门、停车摔倒或 heading 恶化，证明收益窗口很窄。

对 Stage285-s2624 fixed trace 的离线 ONNX 消融显示，原响应分支会产生 hip yaw 与 waist yaw 修正；禁止这些模式后，fast-upper heading 从 `0.433` 降至 `0.328`，说明 yaw 子空间叠加确实是主要干扰源之一。

## 结论

本轮不是完整晋级，但有可复现的正进展：

1. 低维响应上下文能够把 stiff/fixed 从 Stage284 的 `1/3` 提高到 `2/3`，并保持三次停车不摔；
2. response 与 Future-intent 不能在 yaw 子空间自由相加；no-yaw 消融显著改善 hard fast cell；
3. 当前 PPO 只训练持续行走，没有训练“行走末态 → brake/stand 交权”，因此一次匹配微调虽把 heading 压进门内，却造成停车摔倒；
4. 既有低速 yaw recovery 在该条件下不可直接复用。

因此 Stage286 是研究甜点位和下一轮基线，但不是可晋级 checkpoint；Stage287/288 淘汰，长训仍锁定。

## 下一步

下一次有效实验应补齐训练任务本身，而不是继续扫 adapter：在同一 episode 中加入速度下降与停止交权片段，让价值函数同时看到行走 heading 和停车生存；响应分支继续禁止 yaw。最小对照为：

1. Stage286 冻结推理基线；
2. no-yaw 响应分支 + 持续行走训练；
3. no-yaw 响应分支 + 行走/减速/停止混合训练。

只有第 3 组同时通过 stiff/fixed 与 stiff/fast 多次复验，才扩展到完整 18-cell 矩阵。
