# 第五轮预注册：有界上肢安全 Adapter

日期：2026-07-28  
状态：待执行；零训练

## 假设

第四轮找到的 `scale=0.25, time_scale=0.5` 不是只对单条 wave 有效的偶然
超参数，而是可以抽象为控制契约：上层允许表达动作意图，但在进入 X2 PD
目标前统一限制幅值、速度，并在身体进入危险状态时平滑退回默认上身。

## 干预

冻结 Stage208-s2550 下层，保持腿 12DOF + 腰 3DOF 的独占权。上肢 Adapter：

- reference scale：`0.25`；
- reference time scale：`0.5`；
- 每关节相对默认姿态最大偏移：`0.12 rad`；
- 每关节目标最大速度：`0.20 rad/s`；
- root tilt 超过 `0.35 rad` 或 root height 低于 `0.58 m` 时，按同一速度限制
  平滑退回默认上身；
- 头保持锁定。

不改 checkpoint、奖励、下层动作、执行器参数、碰撞模型或 heading controller。

## 动作集合

1. 实机采集 `wave`：长时、非周期上肢动作；
2. ACCAD `Swing_Arms_While_Stand`：轻度双侧摆臂；
3. BMLrub `knocking1`：中度单侧动作；
4. ACCAD `Box_lift`：大幅双侧压力测试，预期会触发幅值裁剪。

只有 14 个肩/肘/腕关节被读取，数据中的 root、腿和腰不进入控制。

## 条件矩阵

- nominal+delay；
- `vx=0.30 m/s, seed=42`；
- `vx=0.30 m/s, seed=7`；
- `vx=0.20 m/s, seed=42`；
- 每组 8 秒（400 control steps）。

共 4 motions × 3 conditions = 12 个候选；每个 condition 使用独立的静态上身
control。

## 单条门禁

- control 与候选均走满 8 秒；
- 上肢 target excursion RMS `>=0.01 rad`；
- target excursion max `<=0.1205 rad`；
- target velocity max `<=0.205 rad/s`；
- 上肢 tracking p95 `<=0.35 rad`；
- 相对同条件 control：heading 恶化 `<=0.15 rad`、lateral 恶化 `<=0.15 m`；
- tilt `<=0.45 rad`；
- 双脚抬升均 `>=20 mm`；
- 危险状态占比 `<=2%`。

## 晋级与停止

- 12/12 通过：解锁“冻结 SONIC 上肢 → 同一 Adapter → Stage208 下层”的真实组合；
- 10–11/12 通过且无跌倒：保留为候选，先分析动作类别边界；
- 少于 10/12 或出现候选跌倒：不接 SONIC，转向下层 yaw/lateral 扰动鲁棒性；
- 不通过时禁止靠继续扫幅值、速度或 PPO 轮数掩盖。

