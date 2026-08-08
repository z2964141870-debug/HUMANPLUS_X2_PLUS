# Stage280：Future-intent × PD × 上肢扰动 × 停车交权阶段裁决

## 结论

Stage267 的 20-update Future-intent 候选配合 1 秒 `brake→stand` 平滑交权，已把官方 AimDK X2 MuJoCo 最难条件中的停车倒地从不稳定状态修复为 **3/3 停车门通过**；完整门仍为 **2/3**，唯一剩余失败是快速摆臂下最坏航向 `0.3124 rad`，超过固定门槛 `0.3000 rad`，因此尚不解锁长训。

## 假设

1. Future-intent 小适配器可在冻结 Stage219 actor 的前提下吸收上肢扰动与 PD 幅值变化。
2. 原 `policy→policy` 瞬时停车失败，不完全是 locomotion actor 本身失败，而是制动状态没有平滑进入 stand backend 的吸引域。
3. 若采用显式速度制动并平滑交权，停车问题应与行走航向问题分离。

## 干预

- 从 Stage219 `model_2600.pt` 启动；8 个 actor 张量逐位冻结，最大差值 `0.0`。
- 只训练 8-mode 腿腰 Future-intent adapter；训练域加入 Kp/Kd `0.9–1.2` 随机化和六类上肢动作。
- 导出 Stage267 `model_2620.pt` 为 121D→15D ONNX：
  - ONNX SHA256：`d1de91fa43b915ffa8f7ec83c0d860c40b1b231fe7122c4690b9960022eb9413`
  - PyTorch/ONNX 最大误差：`1.0431e-7`
  - 零上肢意图与零前向命令均严格退回基础 actor，误差 `0.0`。
- 停车采用：速度闭环制动 `K=1.5`，速度降至 `0.10 m/s` 后，在 1.0 秒内平滑混合到 50% stand backend。
- 所有评估门槛保持不变。

## 对照

| 对照 | 最难条件结果 | 解释 |
|---|---|---|
| Stage219 baseline，soft PD + fast upper | 既有矩阵仅 14/18，最难单元反复航向越界/停车倒地 | 原复合门未闭合 |
| Stage267 25 update，policy stop | 航向越界且停车倒地，`z≈0.10 m` | 继续训练会破坏适配器 |
| Stage267 20 update，瞬时 policy stop | 行走偶尔通过，但停车倒地 | 交权状态不在 stand 吸引域 |
| `brake_then_policy` 瞬时接管 | 不再普遍倒地，但停车漂移约 `0.152–0.155 m` | 方向正确，裕量不足 |
| 0.5 s `brake→stand` 混合 | 停车 2/3 | 接管冲击降低，但仍临界 |
| **1.0 s `brake→stand` 混合** | **停车 3/3，完整门 2/3** | 当前甜点 |
| 已验证 heading-action recovery 叠加 | 航向峰值恶化到 `0.3245 rad` | 该组合淘汰 |

## Stage278 严格结果

| run | full | move | stop | heading max | stop drift | stop z min | settle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | FAIL | FAIL | PASS | 0.3124 | 0.1404 m | 0.6167 m | 3.42 s |
| 2 | PASS | PASS | PASS | 0.2998 | 0.1349 m | 0.6154 m | 2.90 s |
| 3 | PASS | PASS | PASS | 0.2974 | 0.1453 m | 0.6147 m | 1.86 s |

权威机器可读结果：`stage278_hard_gate_summary.json`。

## 结论

- **已解决到可证实程度：** soft PD + fast upper 下的停车生存、漂移和稳定时间，3/3 通过。
- **未解决：** 行走阶段航向只有约 `0.0124 rad` 的负裕量，完整复合门未达到 3/3。
- **训练行为：** 20 update 优于 25 update，说明该适配器存在早期峰值；不能用训练 reward 单独选择 checkpoint。
- **长训状态：** 仍锁定。当前成果是拆清了“停车交权”和“行走航向”两个问题，而不是宣称整套后端完成。

## 下一步

1. 固定 Stage278 停车契约，不再扫 brake/stand 参数。
2. 只针对 soft PD + fast upper 的行走航向增加训练裕量，候选优先为更小学习率、官方 hard-gate early stopping、显式 yaw/lateral margin reward。
3. 新候选先过最难单元 3/3；通过后才运行 `PD {0.9,1.0,1.2} × upper {fixed,fast} × 3` 完整矩阵。
4. 完整矩阵全通过后，才解锁更长训练和 turn/stop 扩展门。
