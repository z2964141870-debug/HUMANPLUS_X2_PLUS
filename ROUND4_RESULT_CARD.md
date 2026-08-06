# 第四轮结果卡：X2 原生下层 × 上肢意图接口

日期：2026-07-28  
状态：完成；达到一个可复现甜点位，但尚未接入真实 SONIC 上肢

## 一句话结论

上下半身分层本身可行：冻结的 X2 原生腰腿策略可以在物理仿真中边走边执行
小幅上肢动作；原速动作在 nominal+delay 中引起的侧漂，可通过只放慢上肢
意图时间轴消除，但当前安全工作区仅验证到 `scale=0.25, time_scale=0.5`，
`scale=0.50` 仍超出下层的航向扰动承受范围。

## 假设

此前 SONIC/G1 → X2 的全身 reference 同时要求腿、root、手部严格跟踪，
会让平衡目标互相竞争。若改成：

```text
上层：只提供肩/肘/腕意图
下层：X2 原生策略独占腿 12DOF + 腰 3DOF
```

则上肢动作不应必然破坏步态；若只在执行器延迟域失败，Adapter 应优先修正
时序，而不是重新训练全身动作。

## 干预

- 下层固定为 Stage208-s2550，零训练；
- 腿 12DOF + 腰 3DOF 始终由下层策略控制；
- 肩、肘、腕 14DOF 由 fixed-feet `wave` reference 提供 oracle 目标；
- 头 2DOF 锁定；
- 上肢从默认姿态零跳变启动；
- 依次检查幅值、ideal/filter/delay 域、手臂/腰部延迟反事实和时间轴缩放。

这里的 oracle reference 只验证物理控制契约，不等于已完成 SONIC 组合。

## 对照

所有组固定同一 checkpoint、seed=42、`vx=0.30 m/s`、10 秒、sole12 碰撞、
self-collision off、同一 gait template 和 heading controller。唯一变化是
上肢目标或执行器时序。

`scale=0` 注入器与旧 evaluator 的所有 trace 数组最大差严格为 `0.0`，
因此后续差异不是 hook 改坏了旧控制器。

## 结果

| 执行器域 / 干预 | 10 s | upper target RMS | upper p95 error | progress ratio | vx RMSE | heading max | lateral max | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| delay，静态上身 control | 500/500 | 0 | — | 1.230 | 0.0935 | 0.510 rad | 0.612 m | 对照 |
| delay，0.25×，原速 | 500/500 | 0.0242 rad | 0.0899 rad | 1.193 | 0.0899 | 0.582 rad | 0.915 m | 失败：侧漂 +0.303 m |
| ideal，0.25×，原速 | 500/500 | 0.0242 rad | 0.0807 rad | 1.251 | 0.0962 | 0.452 rad | 0.750 m | 相对门通过 |
| filter，0.25×，原速 | 500/500 | 0.0242 rad | 0.0845 rad | 1.105 | 0.0833 | 0.751 rad | 1.256 m | 相对门通过 |
| delay，腰部零显式延迟，0.25× | 500/500 | 0.0242 rad | 0.0904 rad | 1.106 | 0.0821 | 0.759 rad | 1.232 m | 相对门通过 |
| delay，0.25×，半速 | 500/500 | 0.0197 rad | 0.0814 rad | 1.195 | 0.0849 | 0.491 rad | 0.505 m | **通过** |
| delay，0.50×，半速 | 500/500 | 0.0394 rad | 0.0950 rad | 1.136 | 0.0827 | 0.683 rad | 1.079 m | 失败：幅值越界 |

通过工作点的双脚抬升为 `39.0/39.1 mm`，tilt 最大 `0.210 rad`，没有
flight，也没有提前终止。相对静态上身，半速 0.25× 的速度 RMSE 改善约
9.2%，heading 减少 `0.019 rad`，侧漂减少 `0.107 m`。

手臂零延迟反事实不适合作为正式 A/B：静态上身 control 在 4.78 秒跌倒，
而动态上身候选走满 10 秒。这反而证明闭环存在窄吸引域，不能用单一二元
通过数解释。候选的侧漂与原 delay 候选接近，故 10.5 帧手臂延迟不是已知
侧漂的充分原因。

## 结论

1. “X2 原生下层无法承受任何上肢动作”被推翻。
2. “只有低通响应导致冲突”被推翻；ideal 和 filter 都通过。
3. 原速 0.25× 在 delay 域的退化与腰部 1 帧延迟存在强交互，但不是简单
   的单关节延迟因果；闭环会随微小时序变化进入不同吸引域。
4. 只做时间轴缩放就找到首个 nominal+delay 正工作点，支持
   **control-contract Adapter 应包含时序整形**。
5. 0.50× 半速仍失败，说明当前下层扰动裕量有限；不能直接把任意幅值的
   SONIC 上肢输出接进来。

## 诚实边界

- 只测了 1 条 `wave`、1 个 seed、1 个前进速度；
- 上肢源是 reference oracle，不是冻结 SONIC 的实际输出；
- “通过”是相对静态上身门禁，绝对 lateral drift 仍为 `0.505 m/10 s`，
  距部署级直线行走还有明显距离；
- 当前 Isaac Sim 因 NVIDIA 内核/用户态版本不一致进入 compatibility mode，
  结果可重复且零幅等价，但本轮不声称使用了有效 GPU PhysX 加速。

## 下一步

不直接接裸 SONIC，也不开长训。第五轮应先实现一个有界的上肢安全 Adapter：

```text
冻结 SONIC 上肢输出
        ↓
幅值上限 + 速度上限 + 时间轴整形
        ↓
健康状态/步态相位门控（异常时退回默认上身）
        ↓
冻结 X2 Stage208 下层
```

先在多个上肢片段、速度和 seed 上证明安全 envelope，再替换 oracle 为冻结
SONIC 输出。若仍出现大幅绝对侧漂，则训练对象应是 X2 下层的 yaw/lateral
扰动鲁棒性，而不是恢复全身 strict reference。

## 证据

- 预注册：`ROUND4_PREREGISTRATION.md`、`ROUND4B_PREREGISTRATION.md`
- 运行脚本：`scripts/run_stage4_upper_disturbance.sh`
- 注入契约：`src/cwi_x2/upper_motion_contract.py`
- 运行时 hook：`hooks/sitecustomize.py`
- 分析器：`tools/analyze_stage4_upper_disturbance.py`
- 原始 JSON/NPZ：`reports/stage4_upper_disturbance/`

