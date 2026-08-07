# stand_backend_balance_i50_zero_pd40_scale2 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_01-04-30_stage_stand_backend_balance_i50_v1/model_2650.pt`
- 生存：`147/500` steps，first done=`147`
- 前进位移/期望：`-0.2503/0.0000 m`，ratio=`None`
- body vx mean/RMSE：`-0.1055/0.2249 m/s`
- yaw rate mean-abs/RMS：`0.2158/0.2594 rad/s`
- heading final/max deviation：`-0.6580/0.6580 rad`
- 最大横漂/倾斜：`0.2420 m` / `0.7824 rad`
- 单支撑/双支撑/腾空比例：`0.041/0.959/0.000`
- 左右脚接触切换：`2/1`
- 迟滞左右脚切换：`2/1`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.240/0.020 s`
- phase 左/右/同时接触匹配：`0.000/0.000/0.000`
- 左右脚最大抬升：`0.0155/0.0308 m`
- action 饱和率：`0.46712`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.0000` / `0.00000`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
