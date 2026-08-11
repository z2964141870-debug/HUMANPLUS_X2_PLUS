# X2 Phase16：放开摆脚后的局部单支撑可达性

日期：2026-08-11  
状态：**0/175 LOCALLY REACHABLE / SUPPORT FOOT MUST RE-PLACE / 0 PHYSICS**

## 问题

Phase15 收窄足距后双支撑覆盖已成立，但原单支撑仍全部失败。本阶段允许摆脚自由移动，只固定候选支撑脚，判断是否能通过重新选择左/右支撑窗口，在小幅 lower15 correction 内把 COM 移到支撑足。

## 合同

每帧、每个候选支撑侧独立线性化：

```text
support-foot centroid xyz correction = 0
COM xy correction = support-foot centroid - current COM
swing foot = unconstrained
```

- 初值：Phase15 hip-roll normalized q。
- 变量：name-mapped lower12+waist3。
- 门：最大关节改动 `<=0.35 rad`、无越限、线性任务残差 `<=1e-6 m`。
- 要求至少存在100ms连续可达的同侧支撑窗口。
- 这是局部线性运动学证书，不是完整轨迹或动力学证书。
- 0 `mj_step`、0 optimizer、0 GPU。

## 结果

| 指标 | 左支撑 | 右支撑 |
|---|---:|---:|
| 可达帧 | **0/175** | **0/175** |
| 最容易帧最大关节改动 | 0.747 rad | 1.433 rad |
| 最大改动 p50 / p95 | 1.277 / 2.810 rad | 1.829 / 2.405 rad |
| delta RMS p50 | 0.445 rad | 0.693 rad |
| 有关节越限的帧 | 100% | 100% |

任一侧可达帧为0，最长连续窗口自然为0ms。

## 裁决

`LOCAL SUPPORT-SELECTION REPAIR REJECTED`

即使完全放开摆脚，固定当前任一支撑足仍无法在合理lower15邻域内完成COM转移。这排除了“只需要换一个更合适的左/右支撑窗口”的解释。

下一生成器必须允许候选支撑足本身重新落位，形成真正的多阶段接触规划，例如：

1. 在双支撑或另一侧支撑时重放置未来支撑足；
2. 同时调整root/COM路径；
3. 再进入目标单支撑；
4. 最后由raw physics验证GRF/contact/平衡。

这已经超出局部residual bridge的表达能力，更接近完整DDR/SBTO/contact-implicit shooting或学习式physical generator。当前不进入physics/teacher/RL。

## 产物

- `audit_phase16_local_support_reachability.py`
- `prereg_phase16_local_support_reachability.json`
- `phase16_result.json`
- 本报告
