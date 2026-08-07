# SONIC/Any2Any → X2 训练阶段复盘（2026-07-12）

## 一句话结论

迁移链路已经正确接通，并找到了能在 X2 标称响应域同时改善 root/body/wrist/foot 的保守候选；但 fixed16 仍有 64.95% EE 早停，不能宣称“迁移完成”或“可上真机”。

## 🎮 任务结算

- [x] 31DOF X2 body/dof/action mapping、头部锁定和 1062 维观测链路通过。
- [x] clean291 PHUMA/X2 参考动作、session03+04 readonly 执行器响应已接入。
- [x] 8 维 `[delay/64, alpha]` 执行器上下文有真实反事实收益。
- [x] 完成 300 轮平衡域、200 轮姿态风险、200 轮终止惩罚、1000 轮长训和 400 轮 EE 专项。
- [x] 每条路线都用 seed4242/fixed16/确定性动作均值做理想域+标称 X2 域验收。
- [x] 保守候选导出 ONNX（1062→31）并通过 100 帧 MuJoCo PD 闭环 smoke。
- [ ] 未通过广泛 motion panel / 高 EE 成功率门，禁止上真机。

## 当前保守候选

`/home/humanplus/x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_phuma_response_context_termpen10_from_bal50_teacher200/model_step_000200.pt`

选它而不是 reward 更高的长训 step400，因为当前目标优先级是“仿真动作不能明显错”，而不是追求更高的标量 reward。

### 标称 X2 执行器域 fixed16

| 项目 | 训练前 | 保守候选 | 结果 |
|---|---:|---:|---|
| reward | 0.4581 | 0.4660 | 改善 |
| mean length / 120 | 15.01 | 14.38 | 退化 |
| root position | 38.3 mm | 33.6 mm | 改善 |
| root rotation | 0.1731 | 0.1643 | 改善 |
| body position | 67.4 mm | 61.5 mm | 改善 |
| left/right wrist | 170.9 / 177.7 mm | 163.5 / 166.8 mm | 改善 |
| anchor-orientation termination | 4.37% | 5.26% | 小幅退化 |
| EE termination | 70.73% | 64.95% | 改善 |
| foot termination | 2.45% | 1.93% | 改善 |

### 导出闭环 smoke

- ONNX: `exported/model_step_000200_decoder.onnx`
- input/output: 1062 / 31
- CPU inference p95: 0.749 ms
- MuJoCo root height: 最低 0.650006 m，100 帧后 0.665886 m
- max joint speed: 8.672 rad/s
- max control: 53.100

该 smoke 只证明导出与闭环运行稳定，不证明动作跟踪已收敛。

## 各路线得失

### 1. 执行器响应上下文：有效

正确响应上下文相比故意错置的“理想上下文”，reward 0.482→0.430，length 15.54→14.64，foot termination 0.89%→5.57%。策略确实使用了动力学上下文。

### 2. 25% 理想域混合：不足以保护 root orientation

修正了“连续均匀采样几乎不可能命中 alpha=1”的问题，并固定 25% 环境为 delay=0/alpha=1。整体和足端有改善，但理想域 root-orientation termination 仍约 17–21%，这条方法单独不够。

### 3. anchor orientation 门前风险 -1：失败

标称域的 root-orientation termination 没有下降，并随训练加深而恶化。该路线已淘汰，大权重文件已清理，日志保留。

### 4. 非超时终止惩罚 -10：有效

相比同一起点，标称域 root/body/wrist/foot 都改善，并生成了当前保守候选。这说明之前“撞门后重置但直接代价为 0”是重要缺口。

### 5. 从有效 step200 延长 1000 轮：部分回升，不是全面回升

长训 step400 在标称域达到 reward 0.6085、length 15.43、EE termination 64.01%，证明“训得更久可能在先降后回升”部分成立。但 root-orientation termination 仍为 6.82%，step800 又恶化，所以盲目延长训练不会自动解决多约束冲突。

### 6. EE 门前稠密风险 -1：局部有效，整体失败

最好 EE 点将标称域 EE termination 64.01%→55.89%，left wrist 167.8→149.1 mm；但 root-orientation termination 升到 18.07%。这是清晰的“顾手丢根”，不选。

## 现在真正卡住的地方

1. 不再是 mapping、joint order、地面或执行器上下文接线错误。
2. 主瓶颈是多约束冲突：单一 LoRA decoder 在 root-orientation 稳定和 EE 跟踪之间交换性能。
3. fixed16 平均段长仍只有约 14–15/120，按 SONIC 文档中的成功率/MPJPE 收敛口径，当前远未收敛。
4. 还缺每条 motion 的失败归因与更广的 clean291 正式评估，fixed16 只能做阶段门。

## 下一阶段建议

不建议继续盲目 PPO 长训或再扫一组标量 reward。更合理的下一步是：

1. 先输出 per-motion/per-cause 失败表，分离 root-tail、left/right wrist、foot 和动作类型。
2. 将 root/EE 从单一加权 reward 改为显式约束或分层 curriculum，而不是让两个风险项互相拉扯。
3. 在扩展训练前先做可视化 fixed16 回放，确认 15cm wrist 门是真实跟踪失败，还是 reward/termination link 与 offset 口径仍有偏差。
4. 候选通过更广 panel 之前，不开真机。

## 工程与存储

- 新增了可复现的平衡域训练和双域 checkpoint panel 脚本。
- 回归测试：46 passed。
- 删除失败/被淘汰的大权重，保留日志、有效起点和候选，累计回收约 3.7GB。
