# stage5c_heading025_box_lift_seed42_vx0p20_delay_s400 固定前进命令物理评估

- checkpoint：`/home/humanplus/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_04-12-50_stage208_sole12_selfoff_ideal075_delay025_resume2450_to2600_v1/model_2550.pt`
- 生存：`283/400` steps，first done=`283`
- 前进位移/期望：`1.6673/1.1320 m`，ratio=`1.4728897451933196`
- body vx mean/RMSE：`0.2993/0.1736 m/s`
- yaw rate mean-abs/RMS：`0.3972/0.5624 rad/s`
- heading final/max deviation：`-0.2385/0.5454 rad`
- 最大横漂/倾斜：`0.3022 m` / `0.7723 rad`
- 单支撑/双支撑/腾空比例：`0.459/0.541/0.000`
- 左右脚接触切换：`16/19`
- 迟滞左右脚切换：`16/17`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.300/0.200 s`
- phase 左/右/同时接触匹配：`0.574/0.472/0.239`
- 左右脚最大抬升：`0.0756/0.1919 m`
- action 饱和率：`0.12862`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.01315`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
