# Phase21 Source Segment Attribution

- 纯离线分析：只读取已有NPZ并逐帧`mj_forward`重建模型接触；没有`mj_step`、没有新replay。
- 分段：JOINT prefix、RL切换±100ms、RL内部（切换100ms后）。

| segment | step p95/max | dq p95/max | limit fraction | root-z min/final | flight |
|---|---:|---:|---:|---:|---:|
| joint_prefix | 0.0108/0.1215rad | 0.530/8.937rad/s | 4.479% | 0.066/0.077m | 0.099 |
| mode_boundary_pm100ms | 0.0880/0.2635rad | 4.609/16.396rad/s | 4.516% | 0.077/0.096m | 0.500 |
| rl_internal_after100ms | 0.1149/0.5821rad | 5.872/31.315rad/s | 5.268% | 0.075/0.229m | 0.459 |

## 结论

- step/dq最大门失败不在JOINT或切换边界：JOINT max `0.121rad/8.94rad/s`，边界 `0.263/16.40`，均低于冻结max门；RL内部达到 `0.582/31.31`，发生在 `19.877s` 的 `right_knee_joint`。
- 但JOINT prefix并不物理稳定：root-z在 `7.018s` 跌破0.42m，即首个complete后约1.58s，RL前已倒地。
- 因此prefix不应作为整段计分reference；最多保留首个complete state/command context作初始化seed，再单独选择稳态、物理有效的RL reference。
- q/body replay误差无法分段，因为既有JSON没有保存trace；在不新跑physics的约束下不能声称aggregate q RMSE由prefix、boundary或RL哪一段主导。
- 这次混合模式失败不能归因于WBT/Any2Any。
