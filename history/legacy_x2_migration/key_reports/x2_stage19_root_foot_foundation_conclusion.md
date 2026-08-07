# Stage 19：根—足物理一致性与真实接触验证

## 假设

旧评估把重置后的短窗口与宽松终止门混在一起，可能掩盖了浮动基座实际跌倒；同时，`clean291` 的根—足筛选门槛可能过松，使动力学解码层同时接收互相冲突的 root 与支撑足目标。

## 干预

- 在 IsaacLab 中为左右足分别增加单 body、ground-only 的接触传感器，过滤目标固定为 `/World/ground/terrain/GroundPlane/CollisionPlane`。
- 接触判定采用竖直力 20 N 和 3 帧历史；足底滑移采用最低足底采样点的 `v + ω×r`，八点足底几何法只做交叉校验。
- 对 `clean291` 全量重做 MuJoCo FK 根—足审计。
- 建立两套不提前终止的浮动基座面板：旧 `core6` 压力动作和严格根—足一致的 4 条移动动作。

## 对照

- checkpoint：`warm200`、`interp025`、`step900`。
- 动力学域：ideal；旧 `core6` 另有 nominal 对照。
- 编码器：teleop；旧 `core6` 另做 G1 encoder A/B。
- 统一 action scale 的 1.0、0.5、0.25 A/B。

## 结果

### 1. 接触测量可信

- X2 估计总质量约 41.97 kg，对应重力约 411.7 N。
- 新 ground-only 传感器双足竖直力中位数约 408.1 N，且能读到单足卸载为 0 N。
- 旧 broad net-force 方案曾出现 4–10 kN 假峰值，原因是混入自碰撞，已停用。

### 2. 旧 core6 确认普遍跌倒

三组 checkpoint 在 400 步、不提前终止的旧 `core6` 面板上，base 倾角 p95 约 1.48–1.53 rad；多数移动动作的 base 高度最低降到约 0.08–0.21 m。G1/teleop encoder A/B 均失败，统一缩小下肢 action scale 只能偶然救回单条动作，不能泛化。

### 3. `clean291` 并不严格干净

- 重新审计 291 条动作后，跨动作的 stance slip p95 为 0.3291 m/s。
- 按 stance slip ≤0.10 m/s、支撑段漂移 ≤0.03 m，只剩 90 条。
- 其中唯一 root 速度 p95 >0.30 m/s 的移动动作是 S21（0.4484 m/s）。
- 再加 root 速度 p95 ≤0.30 m/s 后，基础课程剩 89 条。

### 4. 严格移动面板出现可重复改善

`warm200` 在 4 条严格移动动作中有 3 条保持站立：

| motion | base height p05 m | base tilt p95 rad | L/R stance slip p95 m/s |
| --- | ---: | ---: | --- |
| S19 | 0.651 | 0.144 | 0.049/0.090 |
| S33 | 0.655 | 0.159 | 0.076/0.027 |
| Basic Skills | 0.652 | 0.114 | 0.183/0.032 |
| S21 | 0.195 | 1.580 | 1.283/1.083 |

`interp025` 与 `step900` 也在相同三条动作上站立、在 S21 上跌倒。因此改善来自数据/动作条件，而不是偶然挑中某个 checkpoint。

## 结论

1. X2 仿真地基与接触测量现在足以揭示真实跌倒，不再依赖重置后的表面分数。
2. 根—足问题尚未完全解决，但已从“多数移动动作普遍失败”收敛为“低冲突移动动作 3/3 稳定、一个高速 root 离群动作失败”。
3. 更干净的数据有明确效果；旧 `clean291` 的宽松门槛是 Stage18 长训越训越差的重要原因之一。
4. 当前没有证据支持立刻改 LoRA 范围或增加非 Any2Any 奖励；应先做一次不改模型/奖励、仅收紧数据并放缓动力学域难度的忠实 Any2Any 连续课程训练。

## 下一步

- 用 `phuma_x2_strict89_foundation_50fps` 从 `warm200` 连续训练 1000 轮。
- actor 仍只训练 `actor_module.decoders.g1_dyn` LoRA，critic LoRA 同训；不加 oracle、不加 upright/foot-slip 新奖励。
- 训练域采用 session03/session04 mixed response，并保留 50% ideal 环境。
- 训练后按 strict 移动、旧 core6 压力、ideal/nominal 三层复评；只有 base 高度、倾角、足底滑移和 root/腕误差共同改善才晋级高算力扩展课程。
