# X2 Stage103b Final Verdict

## 结论

Stage103b 是一个通过全部预注册离线门的 **OFFLINE_TRAINABLE_CANDIDATE**，不是“已经解决走路”的模型。它修掉了 Stage103 会通过缩短参考路径来改善 root RMSE 的评估漏洞，同时保留了旧的运动学、滑移、动态代理、上肢与摆脚语义门；下一步仍必须用冻结策略做同条件物理 A/B。

## 假设

Stage103 的 whole-clip joint/root 优化方向并非完全错误，因为物理 A/B 中 root XY RMSE 均值从 `0.1914 m` 降至 `0.1515 m`（改善 `20.8%`），并保持 `4/4` 稳定和 `4/4` 足端门。但是它把 D/B 世界水平位移分别从 `0.13687/0.17800 m` 缩短到 `0.09651/0.15081 m`，使 progress 从 `1/4` 变成 `0/4`；contact 仍为 `0/4`。因此，缺失的是世界路径语义约束，而不是再放松旧门。

## 干预

在 Stage103 的同步整段 joint/root 稀疏最小二乘中，只新增两类源世界坐标目标：

- 每帧沿原路径方向的 root progress 残差，权重 `3`；
- 首末帧水平位移向量残差，权重 `30`。

保留 `dynamic_weight=0.1`，且没有放松 Stage103 的任何滑移、加速度、上肢、摆脚、连续性、动态代理或运动学阈值。本阶段未启动 PPO、Isaac rollout 或真机。

## 对照

控制参考始终是未修改的 Stage100-v2，不是已经缩短路径的 Stage103。正式扫描从同一 Stage100-v2 输入重建所有候选，镜像动作由 canonical 精确生成；另用独立脚本重新检查 source hash、路径、镜像与 provenance。

## 结果

选中候选为 `dyn0p1_p3e30`：

| 指标 | 结果 | 门限 |
| --- | ---: | ---: |
| B 位移比例 | `0.9896` | `[0.90, 1.10]` |
| D 位移比例 | `0.9790` | `[0.90, 1.10]` |
| endpoint 相对误差最大值 | `2.51%` | `≤10%` |
| along-path RMSE 最大值 | `9.36 mm` | `≤15 mm` |
| stance slip p95 最大值 | `0.5427 m/s` | `≤0.55 m/s` |
| root accel p95 最大值 | `8.1405 m/s²` | `≤15.25 m/s²` |
| upper error p95 最大值 | `1.25 mm` | `≤15 mm` |
| swing-relative error p95 最大值 | `62.72 mm` | `≤65 mm` |
| dynamic p95 ratio 最大值 | `0.5310` | `≤0.80` |
| dynamic mean ratio 最大值 | `0.5787` | `≤0.85` |
| 标准运动学门 | `4/4 PASS` | `4/4` |

独立验证结果：source hash、canonical/mirror 精确关系、Stage103b provenance、世界路径门、标准 4/4 运动学门均通过；MuJoCo 根-足审计完成，slip p95 最大值为 `0.542656 m/s`。

与 Stage103 相比，Stage103b 为恢复路径语义付出了小幅动态代理退让：dynamic p95 ratio 从 `0.4593` 变为 `0.5310`，mean ratio 从 `0.5060` 变为 `0.5787`，但仍明显优于源参考的 `1.0` 且通过原门。这是诚实的可解释折中。

## 结论边界

这次只能证明“参考修复不再靠缩短世界路径获利，并通过全部预注册离线门”。LIPM/ZMP/DCM 是 MJCF 模型估计，不是动力学可执行证书；Stage103b 尚未证明 progress 或 contact 的物理门已经改善，更不代表完整 X2 迁移完成。

## 下一步

用同一冻结策略、同一 seed、同一执行器域，对 Stage100-v2、Stage103、Stage103b 做一次三方物理 A/B。只有 Stage103b 在恢复诚实 progress 比较的同时仍保持 `4/4` 稳定和足端门，才将其晋级为后续 Any2Any 长训参考；contact 若仍为 `0/4`，应作为独立的接触时序/策略条件问题处理，不能再通过改短 reference 掩盖。

正式产物：

- `motion_lib_x2/stage103b_official_true_forward4_trajectory_balance_path_v1`
- `docs/reports/x2_stage103b_trajectory_balance_scan.json`
- `docs/reports/x2_stage103b_world_path_gate.json`
- `docs/reports/x2_stage103b_reference_feasibility.json`
- `docs/reports/x2_stage103b_kinematic_gate.json`
