# BASE Phase38：连续成功 suffix 目标审计

## 结论

Phase26 的低维 CEM 不是单纯“搜索预算不够”，而是**目标表示不完整**。它把末端分别拉近了 success-safe 的若干边缘状态，却没有靠近任何一条可连续维持 1 秒的真实成功闭环 suffix。

本阶段只读解析 Phase19-v2、Phase25 的稳健 LOO-p95 标定和 Phase26 最优 bridge；物理步数、optimizer 步数、训练次数均为 0。

## 合同

- success-safe：366 行；critical-from-failure：66 行。
- 连续目标必须来自同一 episode，当前行及后续 50 个 50 Hz tick 全部 eligible + success-safe，共 1.00 s。
- 满足连续合同的候选起点为 215 个。
- 距离沿用 Phase25：逐特征 robust scale，按 success 类内 LOO-p95 归一化；joint-position 与 previous-action 两组等权 RMS，避免 31D q 因维数支配结果。
- Phase19 的 action 只表示 controller history/state，不是 expert imitation label。
- 从 `physical q - obs[12:43]` 恢复的默认姿态跨所有 eligible 行最大误差为 `6.19e-08 rad`，因此没有混用 absolute q 与 q-default。

## 最接近的连续成功目标

- episode：`phase19_stage326_stop_event_stateful_v2_r4`
- source tick：`354`
- stop elapsed：`0.88 s`
- snapshot SHA256：`fb658c3494e253f073a8dcedc88f7a583b1513ae248eeb641fb562fea38d5612`
- controller-state SHA256：`7124140a61827f487dfce5a541300c693b8629baba4caed5e8c3286483eeab3b`
- 后续 51 行逐 tick 连续且全部 success-safe。

相对 Phase26 best endpoint：

| 距离 | 归一化值 | 判定 |
|---|---:|---|
| joint position | 2.804 | 超出 success LOO-p95 |
| previous action | 4.303 | 明显超出 |
| 两组等权 composite | 3.632 | 不在连续成功目标邻域 |

原始量纲 RMSE 为：q `0.1025 rad`，previous-action `0.4922`。最大缺口集中于右膝、右髋 pitch、腰 pitch，以及对应的右膝/腰 pitch/右踝 pitch 动作历史。

更关键的是，即使分别选择单组最邻近目标：

- q 最邻近连续 suffix 仍为 `1.955 ×` success LOO-p95；
- previous-action 最邻近连续 suffix 仍为 `4.209 ×` success LOO-p95。

这解释了 Phase26 看似矛盾的现象：姿态、gravity、root 和短时安全都有改善，但 recovery 接管后 q/history 很快离域。旧 cost 只要求每组分别靠近 success 集合中的某个最近邻；这些最近邻未必属于同一时刻、同一 episode，更不保证有连续未来，因此可能形成不可实现的“拼接目标”。

## 裁决

下一次 BASE 实质实验若继续，应冻结上述 51-tick suffix，把完整 physical/controller sequence 作为一个联合目标：显式同步 q、previous/issued action history，并保留 root/gravity/safety 门。它仍需要新的预注册和一次低资源物理验证，当前不解锁训练。

明确停止继续做的事情：在不改变目标表示的情况下，只扩大 Phase26 五个手工 q-mode 的 population、iteration 或随机 seed。现有证据已经说明这种搜索即使 cost 下降，也不保证进入一个真实连续的成功闭环盆地。

## 证据边界

- 本报告没有证明 51-tick suffix 能从 Phase26 snapshot 动力学可达。
- controller action 是历史状态，不被冒充成专家动作监督。
- 没有 AimDK ROS rollout、MuJoCo integration、PPO、GPU 或真机。
- 因此状态是 `SEQUENCE_TARGET_IDENTIFIED / PHYSICS_AND_TRAINING_LOCKED`。

## 文件

- 审计工具：`tools/official_x2/audit_phase38_sequence_consistent_bridge_target.py`
- 机器可读结果：`reports/official_x2/phase38_sequence_consistent_bridge_target.json`
- 来源：`manifests/x2_phase19_outcome_aware_state_role.json`
- 对照：`reports/baseline/x2_recovery_phase26_bridge_cem.json`
