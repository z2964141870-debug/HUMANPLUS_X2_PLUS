# stand_backend_scratch_i150_zero_pd40_scale2 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_01-17-25_stage_stand_backend_scratch_resume100_to300_v1/model_150.pt`
- 生存：`500/500` steps，first done=`None`
- 前进位移/期望：`-0.0179/0.0000 m`，ratio=`None`
- body vx mean/RMSE：`-0.0009/0.0117 m/s`
- yaw rate mean-abs/RMS：`0.0758/0.0903 rad/s`
- heading final/max deviation：`0.0363/0.0505 rad`
- 最大横漂/倾斜：`0.0133 m` / `0.0956 rad`
- 单支撑/双支撑/腾空比例：`0.000/1.000/0.000`
- 左右脚接触切换：`0/0`
- 迟滞左右脚切换：`0/0`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`10.000/10.000 s`
- phase 左/右/同时接触匹配：`0.000/0.000/0.000`
- 左右脚最大抬升：`0.0042/0.0034 m`
- action 饱和率：`0.00000`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.0000` / `0.00000`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
