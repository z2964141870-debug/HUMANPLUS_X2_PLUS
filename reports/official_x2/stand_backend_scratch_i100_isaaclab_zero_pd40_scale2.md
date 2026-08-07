# stand_backend_scratch_i100_zero_pd40_scale2 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_01-07-36_stage_stand_backend_scratch_i300_v1/model_100.pt`
- 生存：`99/500` steps，first done=`99`
- 前进位移/期望：`-0.3599/0.0000 m`，ratio=`None`
- body vx mean/RMSE：`-0.1807/0.2938 m/s`
- yaw rate mean-abs/RMS：`0.0323/0.0370 rad/s`
- heading final/max deviation：`0.0533/0.0533 rad`
- 最大横漂/倾斜：`0.0181 m` / `0.7762 rad`
- 单支撑/双支撑/腾空比例：`0.000/1.000/0.000`
- 左右脚接触切换：`0/0`
- 迟滞左右脚切换：`0/0`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`1.980/1.980 s`
- phase 左/右/同时接触匹配：`0.000/0.000/0.000`
- 左右脚最大抬升：`0.0154/0.0198 m`
- action 饱和率：`0.00000`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.0000` / `0.00000`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
