# stage4_upper_scale0p50_delay_s150 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt`
- 生存：`150/150` steps，first done=`None`
- 前进位移/期望：`1.0608/0.9000 m`，ratio=`1.1786295308007135`
- body vx mean/RMSE：`0.3545/0.0916 m/s`
- yaw rate mean-abs/RMS：`0.3319/0.4142 rad/s`
- heading final/max deviation：`-0.1830/0.2341 rad`
- 最大横漂/倾斜：`0.1298 m` / `0.2127 rad`
- 单支撑/双支撑/腾空比例：`0.587/0.413/0.000`
- 左右脚接触切换：`12/11`
- 迟滞左右脚切换：`12/11`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.160/0.220 s`
- phase 左/右/同时接触匹配：`0.543/0.426/0.245`
- 左右脚最大抬升：`0.0359/0.0369 m`
- action 饱和率：`0.12444`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.01378`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
