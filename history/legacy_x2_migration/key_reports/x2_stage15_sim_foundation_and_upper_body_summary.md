# Stage 15：X2 仿真地基与上半身专项

## 游戏任务结算

- [x] 修正 MuJoCo 控制时钟：每个 20 ms policy frame 实际执行完整 physics substeps。
- [x] IsaacLab / MuJoCo 双仿真零动作浮动站立 10 s。
- [x] 建立 fixed-base、upper-body-only 的长挥手语义测试。
- [x] 300 轮肩/肘/腕输出行 LoRA 专项训练，逐 50 轮独立筛 checkpoint。
- [x] 用最终 11IMU/teleop encoder 复测候选。
- [x] 修复强下肢 PD 与 source-G1 action scale 的冲突。
- [x] 完整策略浮动基座 15 s 不倒。
- [x] 生成零动作站立与 reference/baseline/candidate 对比视频。

## 地基门

`foundation_stiff_lower` 对腿、足和腰使用分组 `Kp=300, Kd=20`，上肢保持已知 X2
应用层参数。IsaacLab 零动作 10 s 结果：

- root z：0.6472–0.6522 m；
- 最大姿态误差：0.0864 rad（4.95°）；
- 最大 root 位置误差：0.0238 m；
- 500/500 control steps，无跌倒。

MuJoCo 在修正真实控制时钟后也通过 500/500 control steps。旧的 `Kp=60, Kd=2`
只坚持约 1.26 s，证明此前的“10 s 站立”是时钟诊断错误，现结果不是沿用旧假阳性。

## 上半身专项结果

固定基座、teleop encoder、15 s 长挥手的 torso-relative wrist p95：

| checkpoint | left wrist | right wrist | torso/base ori |
| --- | ---: | ---: | ---: |
| Stage12D baseline | 0.2379 m | 0.2005 m | 0.1399 rad |
| Stage15 step250 | 0.2184 m | 0.1730 m | 0.1237 rad |

step250 相对 baseline 的左右腕分别改善约 8.2% / 13.7%，躯干姿态改善约 11.6%。
step300 虽继续改善右腕，但左腕和躯干回退，因此没有用“最后一轮”冒充最佳轮。

## 完整动作接口修复

单纯把 source-G1 action scale 用到 `Kp=300` 的 X2 腿腰，会让完整策略立刻失稳：root
位置 p95 达 3.9399 m。新增 `foundation_hybrid`：

- 腿/足/腰：按 X2 `0.25 * effort_limit / stiffness` 做 torque-normalized scale；
- 肩/肘/腕：保留 source-G1 scale，维持冻结解码器的上肢输出语义；
- 头部：继续严格锁为零。

修复后 step250 完整动作、浮动基座、teleop encoder 的 15 s 长挥手结果：

- root position p95：0.0099 m；
- root orientation p95：0.0375 rad（2.15°）；
- 无任何终止或跌倒。

在同一最终地基/尺度下，Stage12D baseline 与 Stage15 step250：

| model | root pos | root ori | torso/base ori | left wrist | right wrist |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage12D baseline | 0.0111 m | 0.0406 rad | 0.1039 rad | 0.2710 m | 0.2139 m |
| Stage15 step250 | 0.0099 m | 0.0375 rad | 0.0982 rad | 0.2570 m | 0.2167 m |

新候选改善 root、躯干和左腕，右腕回退 2.8 mm。它已经比旧 baseline 更稳，但动作跟踪仍明显弱于参考，不能宣称迁移完成。

## 肉眼验收视频

- `stage15_visual_foundation/x2_foundation_zero_action_float_10s_annotated.mp4`
- `stage15_visual_foundation/wave_reference_vs_baseline_vs_step250_15s.mp4`

三栏视频按同一时间轴显示 reference、旧 Stage12D baseline 和 Stage15 step250。视频是 trace
中保存的真实 IsaacLab robot/reference qpos 经 MuJoCo 运动学渲染，不是重新运行或人工修饰动作。

## 下一关

地基已通过；主要剩余问题从“开局倒地”收敛为“稳定但手臂跟踪幅度/姿态不足”。下一阶段应在
`foundation_hybrid` 上继续做上肢动力学解码适配和长动作验证，而不是再改地面或盲目刷全身训练。
