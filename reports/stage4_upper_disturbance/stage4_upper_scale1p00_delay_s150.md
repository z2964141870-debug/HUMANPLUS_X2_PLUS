# stage4_upper_scale1p00_delay_s150 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt`
- 生存：`150/150` steps，first done=`None`
- 前进位移/期望：`1.0426/0.9000 m`，ratio=`1.1584565374586318`
- body vx mean/RMSE：`0.3517/0.0918 m/s`
- yaw rate mean-abs/RMS：`0.3689/0.4752 rad/s`
- heading final/max deviation：`-0.2908/0.3465 rad`
- 最大横漂/倾斜：`0.1902 m` / `0.2095 rad`
- 单支撑/双支撑/腾空比例：`0.587/0.413/0.000`
- 左右脚接触切换：`12/11`
- 迟滞左右脚切换：`12/11`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.180/0.240 s`
- phase 左/右/同时接触匹配：`0.543/0.436/0.255`
- 左右脚最大抬升：`0.0375/0.0424 m`
- action 饱和率：`0.12444`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.00978`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
