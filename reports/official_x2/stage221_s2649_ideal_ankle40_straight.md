# stage221_s2649_ideal_ankle40_straight 固定前进命令物理评估

- checkpoint：`/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/logs/rsl_rl/x2_lower_velocity_flat/2026-08-08_05-14-59_stage221_official_ankle40_anchor001_resume2600_to2650_v1/model_2649.pt`
- 生存：`200/500` steps，first done=`200`
- 前进位移/期望：`0.5558/1.2000 m`，ratio=`0.4631330072879791`
- body vx mean/RMSE：`0.1338/0.2503 m/s`
- 横向位移/期望：`0.1052/0.0000 m`
- body vy mean/RMSE：`0.0459/0.1326 m/s`
- yaw rate mean-abs/RMS：`0.3171/0.3888 rad/s`
- heading final/max deviation：`-0.2712/0.4152 rad`
- 最大横漂/倾斜：`0.1107 m` / `0.7799 rad`
- 单支撑/双支撑/腾空比例：`0.135/0.865/0.000`
- 左右脚接触切换：`6/22`
- 迟滞左右脚切换：`4/22`（30 N on / 5 N off）
- 迟滞模式中位持续时间：`0.320/0.080 s`
- phase 左/右/同时接触匹配：`0.576/0.584/0.208`
- 左右脚最大抬升：`0.0519/0.0501 m`
- action 饱和率：`0.13367`
- action scale 评估倍率：`2.000`
- sagittal scale 评估倍率：`1.000`
- gait template/scale：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_official_forward_gait_phase_template_15dof.npz` / `0.150`
- template bias max / induced clip：`0.2696` / `0.01360`
- actor 左右镜像 RMSE/max：`None` / `None`

该报告只陈述物理轨迹，不以训练 reward 单独判定会走。
