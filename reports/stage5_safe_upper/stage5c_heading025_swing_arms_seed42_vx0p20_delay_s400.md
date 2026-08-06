# stage5c_heading025_swing_arms_seed42_vx0p20_delay_s400 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt`
- 生存：`284/400` steps，first done=`284`
- 前进位移/期望：`1.6437/1.1360 m`，ratio=`1.4469525973561785`
- body vx mean/RMSE：`0.2936/0.1742 m/s`
- yaw rate mean-abs/RMS：`0.4070/0.5709 rad/s`
- heading final/max deviation：`-0.2054/0.5265 rad`
- 最大横漂/倾斜：`0.2956 m` / `0.7912 rad`
- 单支撑/双支撑/腾空比例：`0.458/0.542/0.000`
- 左右脚接触切换：`16/19`
- 迟滞左右脚切换：`16/19`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.300/0.200 s`
- phase 左/右/同时接触匹配：`0.574/0.438/0.222`
- 左右脚最大抬升：`0.0772/0.1940 m`
- action 饱和率：`0.12770`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.01123`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
