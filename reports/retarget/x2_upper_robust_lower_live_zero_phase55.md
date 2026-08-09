# WBT Phase55 — standard-93D upper-robust lower live zero contract

## 假设

Stage219 的标准 `93D → 15D` 下层 actor/critic 可以在不改变 source 数值语义的前提下接收外部上肢扰动课程：policy 仍只输出腿12+腰3；上肢14仅由外部有界 target adapter 驱动，但上肢实际 `q/dq` 仍保留在 93D proprioception；头2保持 nominal。

## 干预

- 以 Stage219 `model_2600.pt` 初始化 actor/critic，checkpoint SHA256 为 `abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb`。
- 接入 Phase50 portable upper artifact（SHA256 `71db36d0206c44da05640f6e3616f918945524e891a5df8051533fcbeb2ab2ef`）。
- 64 env 固定配对：32 env 无上肢扰动，32 env 有界上肢扰动；这是 zero gate 的确定性 sampler，不是训练随机 sampler。
- 上肢 excursion 上限 `±0.12 rad`、slew 上限 `0.20 rad/s`；reset 时 delta 严格为零。
- `std` 冻结；未构造 optimizer，未执行 env control step，未保存 checkpoint。

## 对照

同一个 Stage219 source model 与其 deep-copy candidate，在两个固定 64×93 batch 上逐元素比较 action/value。B=0 时外部 upper adapter 不进入 15D action residual。

## 结果

- live Isaac 初始化与一次 reset 通过：policy/critic observation 均为 `64×93`，action 为 `64×15`，sim articulation 为 31DoF。
- policy/critic term 均严格为 `base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(31), joint_vel(31), actions(15), gait_phase(4)`；因此上肢实际 q/dq 确实保留在 93D。
- action joint order 严格为腿12+腰3；head yaw/pitch default 均为 0。
- sampler 为 32 fixed / 32 active；reset delta max=0，fixed future delta max=0，active future delta max=`0.0658031 rad`，未越界。
- 两个固定 batch 的 action max-abs difference 均为 0；value max-abs difference 均为 0；输出全部 finite。
- candidate parameter hash 前后完全相同：`c8727a088cd98415bd5705cce74ad57c6c51f63c9f0f36b8e88c1ed176934830`。
- trainable names 是 Stage219 actor/critic 全部 dense weights/biases；唯一 frozen parameter 是 `std`。
- optimizer constructed=false；optimizer steps=0；environment control steps=0；checkpoint created=false。

首次 live 尝试被硬门正确拒绝：reset 后旧 hook 会对 50% mask 做 Bernoulli 重采样，无法保证 paired batch 精确 32/32。修复仅增加 zero-gate 专用 deterministic split，不改变 action、physics 或模型语义；修复后通过。

## 结论

**PASS_LIVE_ZERO_UPDATE_ONLY**。Phase54 的 dedicated standard-93D 方向现已闭合到真实 live initialization/reset/fixed-forward：Stage219 source 在 B=0 下 action/value bit-exact，15D lower/waist action boundary、31D realized q/dq observation、external upper14 target 和 nominal head2 均按合同成立。

这不证明 upper-disturbance robustness，不证明训练有效，也不授权多 update；本阶段没有 physics control rollout。

## 下一步

如获授权，只允许按预注册执行唯一 1-update A/B：训练 sampler 使用可复现的 50/50 upper-none/bounded-upper 分布，保护 ordinary source 回归；完成后先回报，不自动进入 5-update。

机器可读证据：`reports/retarget/x2_upper_robust_lower_live_zero_phase55.json`。
