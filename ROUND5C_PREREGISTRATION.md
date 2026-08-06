# 第五轮 C：IMU 航向早期回退

日期：2026-07-28  
状态：待执行；零训练

## 假设

Stage5 的低速跌倒不是毫无预兆：`wave/swing` 的初始航向偏差分别在
`4.00/2.96 s` 超过 `0.35 rad`，而 tilt/height 失效直到 `5.88/5.36 s`
才出现。只看倾斜和高度的回退触发太晚；利用实机已有 torso IMU 的 yaw，
可以在闭环真正失稳前撤回上肢扰动。

## 干预

保持 Stage5 的 `scale=0.25`、`time_scale=0.5`、最大偏移 `0.12 rad`、
最大目标速度 `0.20 rad/s` 不变。只新增：

- 每个 episode 第一次 action 时记录 IMU yaw；
- 最短角差超过 `0.25 rad` 后锁存风险状态；
- 上肢目标以原速度上限平滑退回默认姿态，本 episode 不再恢复；
- tilt `0.35 rad`、height `0.58 m` 的原回退继续保留。

该量只需 torso IMU，不使用仿真真值 foot force、COP 或世界 XY。

## 对照

直接使用 Stage5 已完成的同 motion、同速度、同 seed=42 结果；不重复 seed=7，
因为关闭 domain randomization 后它与 seed=42 逐项完全相同，不构成独立重复。

测试 4 个动作 × `vx=0.30/0.20 m/s`，每条最多 8 秒。

## 门禁

安全：

- 8/8 无提前终止；
- tilt 不超过 `0.45 rad`；
- tilt/height hazard 占比不超过 2%。

效用：

- `vx=0.30` 的 4 条仍满足原相对 heading/lateral/foot/upper 门；
- 航向未越界的动作不得无故回退；
- 航向越界动作必须在越界后平滑撤回，而不是瞬时跳变；
- 单条 target excursion RMS 仍至少 `0.01 rad`。

低速条件允许在风险出现后牺牲剩余 gesture，但不得把“尽早关闭所有上肢”
包装成完整意图跟踪成功。

## 裁决

- 安全与正常速度效用都通过：得到“带 IMU supervisor 的正常行走上肢接口”
  甜点位，可开始冻结 SONIC 上肢的 shadow 接入；
- 只消除跌倒但正常速度效用失败：仅保留为 emergency guard；
- 仍有跌倒：该回退不足，停止接口调参并训练/替换低速下层。
