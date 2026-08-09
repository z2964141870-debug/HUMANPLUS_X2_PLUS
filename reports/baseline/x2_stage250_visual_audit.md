# X2 Stage250 报告视频视觉审计

日期：2026-08-09

## 结论

用户指出的三项现象全部有客观证据支持：旧视频确实严重掉帧，左右转角度太小而不易肉眼区分，机器人行走时也确实存在约 10–11° 的持续后仰。Stage250 的 official full gate 结论仍然成立，但它只能证明基本位移门通过，不能证明动作自然、转向展示清楚或视频质量合格。

## 1. 卡顿来自哪里

三个原始 X11 录像均标称 30 fps、约 21 s，但 `ffmpeg mpdecimate` 只保留 32–33 张不同画面；即 MuJoCo GUI/X11 捕获大量重复帧。对应 official trace 的控制周期 p50 约为 20 ms，因此旧视频的主要卡顿来自 viewer 刷新/录屏链路，而不是 50 Hz 控制循环本身。

这不排除策略动作自身存在 jerk；它只说明旧录像不能用于判断 jerk。为消除该混淆，新增脚本逐帧重放 immutable official 50 Hz 状态，不执行重新仿真或修改轨迹。

## 2. 左右转为什么像直行

| 动作 | 命令 | 实测累计 yaw | 前进位移 | 判断 |
|---|---:|---:|---:|---|
| straight | `wz=0` | `+4.10°` | `1.204 m` | 存在少量航向漂移 |
| right | `wz=+0.15 rad/s` | `+19.37°` | `1.301 m` | 仅浅弧线转向 |
| left | `wz=-0.09 rad/s` | `-17.85°` | `1.259 m` | 仅浅弧线转向 |

右转和左转的符号正确，但只有约 18–19°，同时保持约 0.3 m/s 前进，因此视觉上更像“向前走并稍微拐弯”，不是明显原地转向或 90° 转弯。旧 full gate 对 turn 只检查 yaw progress ratio，并没有要求适合汇报的绝对转角。

## 3. 后仰是否真实

由 trace 中 projected gravity 重建 ZYX pitch，负值表示后仰：

| 动作 | pitch mean | median | p05–p95 |
|---|---:|---:|---:|
| straight | `-11.29°` | `-11.52°` | `-13.15° … -8.06°` |
| right | `-10.20°` | `-10.26°` | `-12.25° … -7.98°` |
| left | `-10.73°` | `-10.94°` | `-12.36° … -7.93°` |

因此“腰部以上往后、后仰前行”不是错觉。现有 root-tilt 门只限制倾斜幅值，不区分前后方向，也没有自然姿态目标；这是门禁遗漏，而不是视频误差。

## 新的权威人工审查视频

- [左右转俯视并排、带轨迹和实测角度](/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/videos/official_x2/report/smooth_trace_replay/x2_turn_direction_official_trace_smooth.mp4)
- [直行侧视、带世界竖直线和实测后仰](/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/videos/official_x2/report/smooth_trace_replay/x2_straight_posture_official_trace_smooth.mp4)

两段均为 50 fps、4.0 s、200 个不同状态帧；素材来自既有 AimDK v1.0 official-physics trace 的离线状态重放，不是新 rollout，也不改变原始结果。

## 决策

1. 旧 X11 合成视频降级为历史记录，不再用于判断动作流畅性。
2. Stage250 保留为 nominal 位移能力基线，但不得表述成“自然转向”或“姿态已解决”。
3. 后续 BASE 门禁必须报告 signed root pitch 的 mean/p05/p95；阈值需先以官方/可接受姿态基准预注册，不能事后按当前候选调门。
4. 报告型转向视频应使用至少 45° 的累计目标或同时展示俯视轨迹；这只是可视化要求，不替代控制门禁。
5. 下一次训练或适配必须把持续后仰作为独立失败形状，不能只优化 survival、速度和 yaw ratio。

机器可读证据见 `x2_stage250_visual_audit.json`。
