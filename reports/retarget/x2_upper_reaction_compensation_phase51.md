# WBT Phase51：training-free 上肢反作用补偿离线裁决

## 假设

Phase50 的上肢 target q/q̇ 会产生可由 official X2 MJCF 计算的 centroidal momentum；若它与 B1−A 的航向或横向扰动在符号、量纲和时序上对应，则可通过一个有界高层 command 前馈补偿，而不直接修改腿12 action。

## 干预

本阶段只有只读模型计算，没有 adapter 改动、physics、训练或 gain 扫描。上肢动量是**官方仿真模型估计**，不是实机力/动量真值。

计算合同：

```text
upper target q/qdot + official MJCF
→ whole-body centroidal P_upper, H_upper with root fixed
→ yaw reaction = -H_upper,z / Izz(q)
→ lateral reaction = -P_upper,y / M
```

## 离线结果

- yaw reaction：范围 [-0.05846, 0.07726] rad/s，RMS=0.01215，4s 积分=0.00262 rad。
- lateral reaction：范围 [-0.000891, 0.001433] m/s，RMS=0.000549，4s 积分=-0.000050 m。
- 实际 B1−A yaw 位移差：-0.27003 rad；横向位移差：-0.06739 m。
- yaw reaction→yaw-rate disturbance 最佳 lag=0.04s，corr=-0.119，R²=0.014。
- lateral reaction→lateral-velocity disturbance 最佳 lag=0.28s，corr=0.226，R²=0.051。

## 唯一预注册 B2（未执行）

- 最小接口：`command_wz`，不是 waist/leg joint residual。
- `wz_ff(t)=clip(-H_upper,z/Izz, ±0.10 rad/s)`；actor command 为 `clip(wz_heading+wz_ff, ±0.10)`。
- 零上肢 target velocity 时 `wz_ff=0`，严格回退 Stage250。
- 不使用 Phase50 回归拟合 gain，不做 time shift，不扫 cap；B2 只允许一条 closed episode。

## 裁决

**STOP_OFFLINE_SIGNAL_DOES_NOT_EXPLAIN_PHASE50_DISTURBANCE**

守恒式信号量纲有界，但不能解释Phase50的系统性漂移：yaw净积分与实际漂移异号且仅约1%，最佳相关也低；lateral信号同样未过方向/R²/净效应门。此时运行B2相当于盲试，按任务停止。

## 下一步

若继续，应先获得独立上肢动作或对称/反对称上肢panel来辨识扰动映射；不能用同一Phase50 episode拟合gain后再在同episode宣称验证。
