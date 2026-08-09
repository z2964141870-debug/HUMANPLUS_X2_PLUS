# X2 速度型下肢控制：报告视频清单

日期：2026-08-09
范围：冻结权重、官方 AimDK v1.0 MuJoCo、无训练、无真机

## 结果

已生成两组相互补充、不能混为一谈的素材：

1. **Capability**：Stage219 权重采用 Stage250 修正后的部署语义，展示 nominal 条件下的站立、起步、直行、左右转和停车；三条原始 rollout 均为 official full-gate pass。
2. **Limitation**：Stage350 matched-event、stiff `1.2×`、固定上肢条件，完整保留横漂、航向失控和停车倒地；official full gate 明确失败。

没有根据画面主观评价动作好坏，视频仅供人工审查。

## Capability 视频

- [MP4：Stage250 nominal 能力](/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/videos/official_x2/report/x2_base_capability_official_stage250_final.mp4)
- [GIF：轻量预览](/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/videos/official_x2/report/x2_base_capability_official_stage250_final.gif)

控制契约：

- domain：`aimdk_x2_v1_official_mujoco`；
- checkpoint：Stage219 `model_2600.pt`，SHA-256 `abcd49a8...f49bb`；
- ONNX：SHA-256 `b95bad36...c0f9`；
- PD：`official_kp_ankle`，Kp/Kd multiplier 均为 `1.0`；
- 控制频率：50 Hz，三条记录的 wall-dt p50 均约 20 ms；
- 上肢：固定；
- gate：straight、right、left 均为 stand/startup/move/stop/full 全通过。

视频为 1280×720、30 fps、41.60 s。只裁掉 MuJoCo viewer 加载画面，物理片段没有变速、插帧或生成动作。GIF 保持原播放时长，但降采样为 640×360、6 fps，因此只作为预览。

## Limitation 视频

- [MP4：Stage350 stiff 失败](/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/videos/official_x2/report/x2_base_limitation_official_stage350_stiff_fixed_final.mp4)
- [原始完整 X11 录制](/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/videos/official_x2/report/x2_base_limitation_stage350_stiff_fixed_raw.mp4)

控制契约：

- domain：`aimdk_x2_v1_official_mujoco`；
- moving checkpoint：Stage306 `model_2652.pt`，SHA-256 `fa2cce83...5365`；
- moving ONNX：SHA-256 `da95011f...bc4c`；
- stationary/recovery ONNX：SHA-256 `edb73c7c...7565`；
- PD：`official_kp_ankle`，Kp/Kd multiplier 均为 `1.2`；
- 控制频率：50 Hz，本次 wall-dt p50 为 `19.999 ms`；
- 命令：`vx=0.30 m/s`、固定上肢、起步—直行—停车 matched event；
- gate：stand 通过，startup、move、stop、full 均失败。

限制案例的量化结果：横向位移 `0.550 m`、最大航向偏差 `0.705 rad`、停车阶段最低 root-z `0.134 m`、最大 tilt `1.537 rad`。这些数值只用于说明为什么它被标成失败，不替代视频本身。

报告 MP4 为 1280×720、30 fps、14.40 s，只删除 viewer 加载画面，倒地及其末段没有裁掉；22.70 s 原始 X11 文件同时保留以便追溯。

## 假设 / 干预 / 对照 / 结论

- **假设**：报告必须同时呈现已验证 nominal envelope 和严格 stiff 失败，不能只选成功片段。
- **干预**：复用三条 immutable Stage250 视频，只新录一次冻结 Stage350 limitation；仅加契约和命令文字。
- **对照**：合成前逐条检查 official JSON；capability 必须 `full_gate_pass=true`，limitation 必须 `false`，否则脚本失败退出。
- **结论**：素材诚实支持“nominal 基本运动可用、stiff fixed-upper matched event 尚未解决”；不支持 WBT 或 stiff-domain 已晋升的说法。

机器可读证据见 [JSON manifest](/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/reports/baseline/x2_base_report_video_manifest.json)。
