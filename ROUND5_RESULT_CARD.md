# 第五轮结果卡：有界上肢 Adapter 的适用边界

日期：2026-07-28  
状态：完成；达到“可裁决的接口边界”甜点位，不解锁真实 SONIC 注入

## 一句话结论

幅值、速度和时间整形可让冻结 X2 下层在 `0.30 m/s` 下承受 4 类动作中的
3 类，但不能覆盖真实 `wave` 或 `0.20 m/s` 低速；更重要的是，发现危险后
再收回手臂反而把低速生存从 `2/4` 降为 `0/4`，因此下一步必须让下层提前
看到上肢意图并学习前馈协调，不能再靠事后 safety filter。

## 假设

第四轮的单条 `wave` 工作点若不是偶然吸引域，则将它固化为：

```text
0.25× 幅值 + 0.5× 时间轴
+ 0.12 rad 最大偏移
+ 0.20 rad/s 最大目标速度
+ tilt/height 健康回退
```

应能在多个动作、速度和随机种子上保持下层步态。

## 干预

- 冻结 Stage208-s2550 下层，不训练；
- 腿 12DOF + 腰 3DOF 继续由下层独占；
- 上层只控制肩、肘、腕 14DOF，头锁定；
- 加入偏移、slew rate、关节软限位、tilt/height 回退；
- 评估真实 `wave`、双侧 `swing`、单侧 `knocking`、双侧大幅
  `box_lift`；
- 追加 Stage5C：初始 IMU yaw 偏差超过 `0.25 rad` 后锁存，平滑退回默认
  上身，检验能否在 tilt/height 失效前救援。

## 对照

每个速度使用同一静态上身 control，完整 nominal+delay、8 秒。原计划使用
seed 42/7；复核发现关闭 domain randomization 后两 seed 的 JSON/trace
逐项相同，因此不能将其当两个独立样本。正式判断按 4 motions × 2 speeds
的 8 个唯一条件。

## 结果

### Stage5 有界 Adapter

- 原始表面计数：`6/12`；
- 有效唯一条件：
  - `vx=0.30`：`3/4` 通过，4/4 生存；
  - `vx=0.20`：`0/4` 通过，2/4 生存；
- 正常速度通过：`swing_arms`、`knocking`、`box_lift`；
- 正常速度失败：真实 `wave`，相对 control 的 heading/lateral 分别恶化
  `0.201 rad / 0.223 m`；
- 低速 `wave/swing` 跌倒；`knocking/box_lift` 生存但方向或横漂超门；
- 所有目标最大偏移不超过 `0.120 rad`、最大速度不超过 `0.200 rad/s`，
  上肢 tracking p95 为 `0.074–0.114 rad`，说明失败不是边界器失效或上肢
  跟不上。

### Stage5C IMU 航向回退

- `3/8` 通过，低速生存从 Stage5 的 `2/4` 下降为 `0/4`；
- 低速 guard 在 `1.66–2.56 s` 触发，目标在 `0.10–0.56 s` 内平滑回到
  默认，但约 `5.6–6.5 s` 仍跌倒；
- 原本能走满 8 秒的低速 `knocking/box_lift` 在回退后都于约 5.66 秒
  跌倒；
- 正常速度 `wave` 虽在 4.88 秒回退，heading/lateral 仍恶化
  `0.196 rad / 0.215 m`，没有恢复轨迹。

## 结论

1. “上下半身分层完全不可行”仍被否定：三个不同来源、不同幅度/单双侧的
   上肢动作在正常步速均严格通过。
2. “固定幅值/速度滤波就能形成通用安全接口”被否定。第四轮 `wave` 正例对
   `<=0.01 rad` 的稀疏目标变化敏感，说明 Stage208 存在窄吸引域。
3. seed 42/7 在当前确定性 evaluator 中不是独立重复；以后必须启用真实
   domain randomization 或初态扰动，不能用改 seed 伪造统计置信度。
4. tilt/height 回退太晚，但提前 IMU-yaw 回退也不是解法；重新收臂会成为
   新扰动，且无法撤销已经积累的横向/航向状态。
5. Stage208 虽观察到当前 arm `q/dq`，但看不到未来上肢目标；在 arms
   约 10.5 个控制帧的辨识延迟下，它只能反应式补偿。当前瓶颈已从
   “Adapter 有没有边界”收敛为“下层缺少上肢预告和扰动训练”。

## 下一步

不解锁裸 SONIC，也不继续扫回退阈值。下一轮只验证一个结构性命题：

```text
未来 0.6–1.0 s 的有界上肢意图 + gait phase
                  ↓
      小型 lower coordination adapter
                  ↓
只修正腰/髋的低维、有界 residual
                  ↓
       冻结 Stage208 主体与上肢路径
```

第一对照应是同一 4-motion×2-speed 面板中的：

1. Stage208 reactive-only；
2. Stage208 + 当前 q/dq、但无 future intent；
3. Stage208 + future upper intent；
4. 第 3 组去掉 phase。

只有第 3 组相对 1/2/4 同时减少跌倒和 heading/lateral，才说明改进来自
“预期协调”，而不是又一个容量更大的 residual。

## 诚实边界

- 仍是 oracle reference，不是冻结 SONIC 的实际输出；
- 没有站立/零速下层，`0.20 m/s` 也已显示基座方向鲁棒性差；
- 当前门是相对 control 门，Stage208 自身绝对横漂仍大；
- Isaac Sim 因 NVIDIA 内核/用户态版本不一致处于 compatibility mode；
- 本轮零训练、零新 checkpoint，旧工程保持只读。

## 证据

- 预注册：`ROUND5_PREREGISTRATION.md`、`ROUND5C_PREREGISTRATION.md`
- 主面板：`reports/stage5_safe_upper/stage5_safe_upper_panel_s400.{md,json}`
- 回退面板：`reports/stage5_safe_upper/stage5c_heading025_panel_s400.{md,json}`
- Adapter：`hooks/sitecustomize.py`、`src/cwi_x2/upper_motion_contract.py`
- 单测：`9 passed`
