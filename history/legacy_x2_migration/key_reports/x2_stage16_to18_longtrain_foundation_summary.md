# X2 Stage16–18：长训前地基结算

## 游戏进度

- [x] Stage16：证明当前 X2 reference、action 语义和仿真动力学可以闭环实现。
- [x] Stage17：修复稳定策略“站得住但手臂跟不上”的主要问题。
- [x] Stage17：用最终 teleop encoder 在三条安全动作上做浮动基座长时验证。
- [x] Stage18：关闭 teacher，恢复完整 dynamics decoder LoRA + critic LoRA + PPO。
- [x] Stage18：修复 warm200 critic observation 与 source14 Any2Any 不兼容的问题。
- [x] Stage18：真实完成一次响应随机化 Any2Any optimizer preflight。
- [ ] Stage18：连续 3000 轮正式训练与每 250 轮 checkpoint 晋级评估。
- [ ] 从长训 checkpoint 选出最终候选，重新生成 reference/baseline/candidate 长视频。

结论：**长训前地基 100% 完成；最终迁移模型尚未完成。** 这里的 100% 只表示长训入口经过了物理、语义、数值和视频四类检查，不表示 3000 轮之后一定达到最终遥操作质量。

## Stage16：reference oracle

**假设：** 先前腕端误差可能来自 action 映射错误、reference 在 X2 上物理不可实现，或执行器响应滞后；必须先把这三者分开。

**干预：** 新增 future-reference oracle，使用运行时真实 default pose、action scale、locked-head mask，将未来 X2 关节目标反解成归一化 action，并直接经过 IsaacLab 物理仿真。

**对照：** fixed-base 与 floating-base；lookahead 1–6 帧；不经过策略网络。

**结果：** floating-base 15 s、750 帧无 reset/non-finite；root position p95 0.0133 m，倾斜 p95 0.0489 rad，左右腕 torso-relative p95 0.0648/0.0609 m。lookahead 从 1 增至 6 帧时，左腕 p95 从约 0.126 m 降到 0.075 m。

**结论：** mapping 和 reference 不是当前主故障；上肢约存在 6 个 control frame（约 120 ms lookahead、约 100 ms净补偿量级）的可学习响应滞后。

**下一步：** 只把 future action 当 rollout supervision，不能泄漏进 policy observation。

## Stage17：上肢 teacher warm-up

**假设：** 在已验证的浮动站立地基上，只校正肩、肘、腕输出并学习提前量，可以改善手臂，而不必先扰动腿腰。

**干预：** fixed-base、upper-body-only、PPO 系数为 0；用 lookahead=6 的 oracle action 对最后一层 14 个上肢输出行做 200 轮监督 warm-up。随后 teacher 完全关闭做独立评估。

**对照：** Stage15 baseline、warm100、warm200；g1 encoder 与最终 teleop encoder；fixed 与 floating。

**结果：** teacher loss 从首段约 2.116 降到最终 0.127。最终 teleop 三动作 floating panel 每条 15 s、共 2250 行，无 done/reset：root position p95 0.0131 m、root orientation p95 0.0605 rad、左右腕 torso-relative p95 0.0911/0.0775 m。长挥手 teleop baseline→warm200 的腕端 p95 为 0.2570/0.2167→0.0996/0.0614 m。

**结论：** root/foot 地基没有被上肢修复破坏，腕端主瓶颈获得明确改善；三条动作均通过，排除了只记住一条 wave 的最简单解释。但它仍不是广动作 Any2Any 模型。

**下一步：** 将 warm200 残差精确合入 base，teacher 关闭，重新开放完整 dynamics decoder。

## Stage18：Any2Any 长训入口

**假设：** warm200 actor 可以作为标准 Any2Any 的更好起点，但其 full33 critic observation（1907 维）不能直接用于 source14 Any2Any（1745 维）。若静默重建 critic 输入层，会给 PPO 引入不必要的高方差。

**干预：** 组合 Stage17 warm200 actor 与既有 faithful Any2Any-1000 的 source14 critic；加载时把两者 LoRA 精确合入 base，再注入 7 层全 dynamics decoder actor LoRA 和 7 层 critic LoRA。oracle、output mask、constraint、retention 均关闭；使用正常 PPO、PHUMA clean291、foundation PD/action scale、session03/04 mixed response。

**对照：**

- 直接 warm200：critic 仅 14/17 张量兼容，输入层 1907→1745 不匹配；value loss 14.514，critic gradient norm 31.641。
- 组合 seed：critic 17/17 完整加载；value loss 0.595，critic gradient norm 1.839。

**结果：** response-randomized smoke 正常完成 1 rollout iteration 和 12 次 optimizer step；actor/critic 梯度均有限；decoder replay 的 mean、sigma、logprob、KL 边界误差全部为 0；退出状态为 0。

**结论：** 长训入口不是仅能启动，而是满足：warm actor 行为保留、critic 维度兼容、Any2Any 参数位置正确、PPO replay 精确、真实 optimizer 可更新。

**下一步：** 运行 `scripts/run_x2_stage18_any2any_from_warm200.sh long`，连续 3000 轮，不以早期 100–200 轮下降判死；每 250 轮保存，最终按 teleop safe panel、clean291 held-out、root/foot/腕端和长视频共同选 checkpoint。

## 关键资产

- 最佳长训前 actor：`logs/ppo_dryrun/x2_stage17_upper_oracle_warm200_cont_v1/model_step_000100.pt`
- Stage18 组合 seed：`checkpoints/x2_stage18_warm200_actor_faithful1000_critic_seed.pt`
- Stage18 启动脚本：`scripts/run_x2_stage18_any2any_from_warm200.sh`
- seed 组合报告：`docs/reports/x2_stage18_seed_composition.json`
- 三栏肉眼对比视频：`docs/reports/x2_stage17_visual_comparison/reference_vs_baseline_vs_warm200.mp4`
- 三动作最终 teleop 报告：`docs/reports/x2_stage17_float_panel3_warm200_teleop.md`
