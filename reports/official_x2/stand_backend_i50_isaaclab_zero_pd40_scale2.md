# stand_backend_i50_zero_pd40_scale2 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_00-57-22_stage_stand_backend_stage208_i50_v2/model_2600.pt`
- 生存：`165/500` steps，first done=`165`
- 前进位移/期望：`-0.2556/0.0000 m`，ratio=`None`
- body vx mean/RMSE：`-0.0697/0.2036 m/s`
- yaw rate mean-abs/RMS：`0.0945/0.1755 rad/s`
- heading final/max deviation：`-0.5760/0.5760 rad`
- 最大横漂/倾斜：`0.0505 m` / `0.7709 rad`
- 单支撑/双支撑/腾空比例：`0.067/0.933/0.000`
- 左右脚接触切换：`0/1`
- 迟滞左右脚切换：`0/1`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`3.300/0.220 s`
- phase 左/右/同时接触匹配：`0.000/0.000/0.000`
- 左右脚最大抬升：`0.0088/0.0305 m`
- action 饱和率：`0.46586`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.0000` / `0.00000`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
