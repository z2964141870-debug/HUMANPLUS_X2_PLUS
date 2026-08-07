# stage213_i2600_biasn040 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_02-14-00_stage213_official_latmatch_bias040_mix050_resume2550_to2651_v1/model_2600.pt`
- 生存：`145/300` steps，first done=`145`
- 前进位移/期望：`1.0302/0.8700 m`，ratio=`1.184087375114704`
- body vx mean/RMSE：`0.3588/0.2287 m/s`
- yaw rate mean-abs/RMS：`0.4251/0.5283 rad/s`
- heading final/max deviation：`0.5695/0.5695 rad`
- 最大横漂/倾斜：`0.1386 m` / `0.7879 rad`
- 单支撑/双支撑/腾空比例：`0.241/0.759/0.000`
- 左右脚接触切换：`6/12`
- 迟滞左右脚切换：`6/12`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.240/0.120 s`
- phase 左/右/同时接触匹配：`0.652/0.416/0.213`
- 左右脚最大抬升：`0.0590/0.0758 m`
- action 饱和率：`0.15632`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.01963`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
