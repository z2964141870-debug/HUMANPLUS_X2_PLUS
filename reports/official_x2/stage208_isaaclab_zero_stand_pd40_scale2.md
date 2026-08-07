# stage208_zero_stand_pd40_scale2 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt`
- 生存：`124/500` steps，first done=`124`
- 前进位移/期望：`0.5852/0.0000 m`，ratio=`None`
- body vx mean/RMSE：`0.2575/0.4113 m/s`
- yaw rate mean-abs/RMS：`0.5615/0.9148 rad/s`
- heading final/max deviation：`0.8302/0.8302 rad`
- 最大横漂/倾斜：`0.1906 m` / `0.6420 rad`
- 单支撑/双支撑/腾空比例：`0.202/0.798/0.000`
- 左右脚接触切换：`0/3`
- 迟滞左右脚切换：`0/3`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`2.480/0.280 s`
- phase 左/右/同时接触匹配：`0.000/0.000/0.000`
- 左右脚最大抬升：`0.0817/0.0945 m`
- action 饱和率：`0.45806`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.0000` / `0.00000`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
