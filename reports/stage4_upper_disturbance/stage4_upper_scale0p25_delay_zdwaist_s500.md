# stage4_upper_scale0p25_delay_zdwaist_s500 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt`
- 生存：`500/500` steps，first done=`None`
- 前进位移/期望：`3.3181/3.0000 m`，ratio=`1.1060486634572346`
- body vx mean/RMSE：`0.3589/0.0821 m/s`
- yaw rate mean-abs/RMS：`0.2924/0.3988 rad/s`
- heading final/max deviation：`-0.7109/0.7592 rad`
- 最大横漂/倾斜：`1.2323 m` / `0.1841 rad`
- 单支撑/双支撑/腾空比例：`0.660/0.340/0.000`
- 左右脚接触切换：`43/42`
- 迟滞左右脚切换：`43/42`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.180/0.240 s`
- phase 左/右/同时接触匹配：`0.549/0.476/0.317`
- 左右脚最大抬升：`0.0381/0.0411 m`
- action 饱和率：`0.13373`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.01253`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
