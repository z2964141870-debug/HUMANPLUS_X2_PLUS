# X2 WBT Phase29：统一接触约束全轨迹 repair existence test

## 裁决

- 冻结条件下 train Silver：`0/3`；可冻结：`False`。
- 本阶段 physics/PPO/policy-optimizer/checkpoint/真机均为 `0`；另有每条8轮 SciPy LSQR 离线轨迹求解，已逐轮记录。
- contact/FK 是模型估计，不是实机 GRF、COP、足底力或真实接触真值。

## 假设 / 干预 / 对照

- 假设：Phase28 的系统性 stance speed/excursion/timing 失败，可能来自逐帧运动学重定向缺少整段接触一致性，而非数据完全不可用。
- 干预：一个统一稀疏 Gauss–Newton 同时优化每帧 root XYZ、腰腿15DOF和每个 source stance phase 的XY anchor；root XY硬限20 cm。
- 对照：Phase28 official-v1 原产物；Phase3 的短窗CEM/局部接触事件和 Phase8 的碎片窗口首帧anchor＋逐帧DLS均不复用。
- 禁止项：逐clip调参、固定root、删帧/删段、强制双脚接触、physics选参。

## 结果

| motion | original tier | repair tier | rootXY max | q correction max | wall(s) | repair rejects |
|---|---|---|---:|---:|---:|---|
| `AMASS-WALK-001` | Reject | Reject | 0.0325 | 0.2819 | 3.7 | `bronze:joint_step_p95 ; bronze:tracked_keypoint_p95` |
| `PHUMA-LUNGE-R-001` | Reject | Reject | 0.0559 | 0.2866 | 1.2 | `bronze:joint_step_p95` |
| `AMASS-KICK-L-001` | Reject | Reject | 0.0966 | 0.4500 | 2.3 | `bronze:joint_step_p95 ; bronze:joint_step_max ; bronze:tracked_keypoint_p95 ; bronze:tracked_keypoint_max` |

## 零运行失败归因

- `AMASS-WALK-001`：stance speed L/R `1.677/1.734 → 0.596/0.368 m/s`，excursion `0.374/0.312 → 0.065/0.038 m`，但仍未过 `0.10 m/s / 0.03 m`；右脚 timing `0.367 → 0.445 s`、flight `0.189 → 0.412` 和 root acceleration `3.47 → 5.87 m/s²` 反而恶化。joint-step p95 `0.130 → 0.120 rad`、keypoint p95 `0.114 → 0.107 m` 虽改善，仍超 Bronze。
- `PHUMA-LUNGE-R-001`：stance speed `0.318/0.205 → 0.044/0.048 m/s`、excursion `0.106/0.026 → 0.007/0.002 m`、右脚 timing `1.243 → 0.067 s`，keypoint p95 `0.129 → 0.094 m`，说明整段相位锚定对接触几何是明确正信号；但 Bronze joint-step p95 `0.141 rad` 完全不变。逐关节只读归因表明主导者是冻结的 `right_shoulder_pitch_joint`，因此当前“只改腰腿且保持源上肢”的表示不可能修掉这项全身门；同时 root acceleration `3.81 → 5.97 m/s²` 超 Silver。
- `AMASS-KICK-L-001`：stance speed/excursion明显下降，但右脚仍 `0.166 m/s / 0.032 m`，contact timing、flight 和 root acceleration 未过；joint-step 被 `left_hip_pitch_joint` 主导并恶化至 p95/max `0.156/0.209 rad`，keypoint仍远超门。左脚摆动高度没有被牺牲，clearance p50/p95 `0.352/0.724 → 0.450/0.859 m`，动作侧别语义保持。
- 因而这不是“全轨迹思想完全无效”：它能显著修正 stance 锚定和 lunge 接触时序；但当前单一目标函数会用更高 root 加速度/flight 交换接触几何，并且无法越过源上肢自身的 Bronze 跳变。根据预注册门，部分指标改善不能替代 `至少1条 Silver`。

## 结论

统一全轨迹表示在三条固定 train 动作上仍未产生 Silver；按预注册门停止，不扩大权重、动作或参数搜索。这只否定当前生成器/配置，不否定 X2、Any2Any 或闭环策略。

## 下一步

Stop this generator. Do not tune weights per clip or expand the search.
