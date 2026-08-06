# DC-PEFT Phase 2 与 Phase 3-I1 报告

日期：2026-07-28

## 双 critic 初始化契约

Stage152-B scalar critic 的最终 base weight、bias 和 LoRA-B 均等分成两行：

```text
V0(x) = 0.5 × Vscalar(x)
V1(x) = 0.5 × Vscalar(x)
V0(x) + V1(x) = Vscalar(x)
```

- 原 checkpoint SHA-256：`b73c345995c5d468c18d223de96b29fff4cd4866e4d56bb6d5296c4a68540679`
- dual init SHA-256：`5894a8587c44cca69983af5924815dbf99eb7e947ad8ce0c6e5fb765580609b2`
- policy：90/90 tensors bit-identical
- dual value load：45/45，mismatch 0
- optimizer/scheduler state：有意不复制，避免错误形状恢复

## Actor compatibility

相同 ideal rollout、4 env、260 steps 下：

- trace 行数：1040 vs 1040
- 所有数值字段：402,480 个，非零差异 0
- sampled/action 字段：68,640 个，非零差异 0
- 非数值字段差异：0

因此双 critic 的引入在 PPO 更新前不改变 Stage152-B actor 行为。

## 一步训练门

| 指标 | B1 single | E1 semantic dual | N1 shuffled dual |
|---|---:|---:|---:|
| objective reward | 0.550146 | 0.550146 | 0.550146 |
| actor grad preclip | 7.753 | 7.648 | 7.623 |
| critic grad preclip | 5.058 | 1.494 | 1.400 |
| approximate KL | 0.001533 | 0.001511 | 0.001527 |
| replay action RMSE | 0.003399 | 0.003389 | 0.003402 |
| reward contract max error | n/a | 1.49e-08 | 2.24e-08 |

E1 各头的初始诊断：

| head | reward mean | TD std | explained variance | pre-update value loss |
|---|---:|---:|---:|---:|
| locomotion_balance | 0.04948 | 0.13281 | 0.69946 | 0.16888 |
| imitation_upper_regularization | 0.08523 | 0.12845 | 0.72137 | 0.13419 |

注意：双头 value loss 与单头 loss 的绝对数值不能直接当作性能收益，因为 target 分解改变了每头幅度；后续比较以每头 EV/TD 曲线和固定物理评价为准。

## 保存/恢复

E1 I1 checkpoint 重新载入双头模型后：

- value tensors：45/45
- mismatched/missing：0/0
- 8-step Isaac smoke：通过

## 裁决

Phase 2 与 I1 数值门通过，解锁 matched 5-iteration 实验。当前尚无证据声称 semantic dual 优于 single 或 shuffled；一步只排除了实现错误和梯度尺度混淆。
