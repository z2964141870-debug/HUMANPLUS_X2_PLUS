# X2 Stage75–77：长训与摆动脚救援结论

日期：2026-07-15

## 假设

此前短训/受限长训越训越差，可能不是 Any2Any 思路无效，而是 actor 更新被末层 LoRA、`5e-8` 学习率和 source-retention 锁死；恢复完整 dynamics-decoder LoRA 后应出现真实长训上升。若随后出现双脚贴地捷径，连续接触监督和稠密摆动脚高度信号应能恢复 liftoff。

## 干预

- Stage75：完整 dynamics-decoder LoRA，3 epoch × 4 mini-batch PPO，无 retention；root `0.10`、接触 `0.25`。
- Stage76：从 Stage75-s1000 继续，接触提升为 `1.0`，启用连续 `weight_transfer`。
- Stage77：从 Stage76-s1000 继续，新增只在 reference swing 生效的足端高度 Gaussian reward，`std=0.025 m`、weight `1.0`。

## 对照

所有阶段使用相同的 4 条 corrected true-forward X2 motion、相同 X2 PD/action scale、50/50 ideal/nominal 训练域；固定回放统一使用 nominal response `0.01`、4 motion × 260 帧。

## 结果

- Stage75 训练 reward/step `0.0553→0.0711`，root error `0.471→0.290 m`，证明完整 Any2Any 更新能力能形成长训整体上升；但 step2000 接触 duty 达到 `0.970/0.987`，学成双脚贴地捷径。
- Stage76 的连续载荷监督将严格接触门从 `0/4` 提到 `2/4`，但双支撑仍明显过多。
- Stage77-s750 首次同时达到 `stable=4/4`、`contact=4/4`、`foot=4/4`。
- 相对 Stage76-s1000，Stage77-s750 左/右摆动脚 p95 离地高度由 `0.020/0.0139 m` 提升到 `0.0455/0.0333 m`；左脚摆动期中位 Fz 由 `93.3 N` 降至 `16.8 N`。
- Stage77-s750 的 root XY RMSE 为 `0.166 m`，高于 Stage76-s1000 的 `0.087 m`；严格世界前进仍 `0/4`，瓶颈转为 B/B-mirror 方向错误和 D-mirror 位移过量。

## 结论

长训“整体越来越好”的命题已在 Stage75 的训练/冻结结构曲线上得到支持；稠密摆动脚高度项也产生了真实物理改善，而不是只刷 reward。当前不再卡在“完全不敢抬脚”，而是卡在保持 liftoff/接触正确的同时恢复 world-root 前进方向与幅度。

## 下一步

以 Stage77-s750 为唯一 seed，保留接触与高度机制，仅将 root tracking 从 `0.10` 提到 `0.20`。若固定回放能保持 contact/foot/stable 4/4 并降低 root RMSE、修正 B/B-mirror 方向，即可进入更广动作 held-out 与视频验收；否则不继续堆 reward。
