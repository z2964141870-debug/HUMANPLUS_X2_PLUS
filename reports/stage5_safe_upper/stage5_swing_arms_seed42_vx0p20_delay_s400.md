# stage5_swing_arms_seed42_vx0p20_delay_s400 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt`
- 生存：`280/400` steps，first done=`280`
- 前进位移/期望：`1.6944/1.1200 m`，ratio=`1.5128327267510548`
- body vx mean/RMSE：`0.3135/0.2107 m/s`
- yaw rate mean-abs/RMS：`0.4236/0.6008 rad/s`
- heading final/max deviation：`0.3623/0.5238 rad`
- 最大横漂/倾斜：`0.3089 m` / `0.7593 rad`
- 单支撑/双支撑/腾空比例：`0.457/0.543/0.000`
- 左右脚接触切换：`16/17`
- 迟滞左右脚切换：`16/17`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.300/0.200 s`
- phase 左/右/同时接触匹配：`0.560/0.457/0.229`
- 左右脚最大抬升：`0.0847/0.1252 m`
- action 饱和率：`0.12190`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.01091`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
