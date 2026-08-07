# stage213_i2650_biasn040 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_02-14-00_stage213_official_latmatch_bias040_mix050_resume2550_to2651_v1/model_2650.pt`
- 生存：`129/300` steps，first done=`129`
- 前进位移/期望：`0.7688/0.7740 m`，ratio=`0.9933044744092364`
- body vx mean/RMSE：`0.3150/0.2244 m/s`
- yaw rate mean-abs/RMS：`0.4326/0.5894 rad/s`
- heading final/max deviation：`0.4821/0.4957 rad`
- 最大横漂/倾斜：`0.2445 m` / `0.7597 rad`
- 单支撑/双支撑/腾空比例：`0.271/0.729/0.000`
- 左右脚接触切换：`4/10`
- 迟滞左右脚切换：`4/10`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.260/0.160 s`
- phase 左/右/同时接触匹配：`0.593/0.531/0.222`
- 左右脚最大抬升：`0.0619/0.0530 m`
- action 饱和率：`0.21809`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.03179`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
