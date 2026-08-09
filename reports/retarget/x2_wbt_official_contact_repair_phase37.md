# X2 WBT Phase37：official-collision phase-aware contact repair

## 裁决

- **PHASE37_OFFICIAL_COLLISION_SILVER_NOT_FOUND**；corrected tier `Bronze → Bronze`。
- 0 physics integration/PPO/policy optimizer；offline sparse trajectory LSQR固定12轮。
- official contact严格为x2.xml 12 active sole spheres/foot signed distance<=0；未使用10.05mm Bronze容差。

## 假设 / 干预 / 对照

- 假设：只修改root-z与腰腿15DoF的连续全轨迹，可能使冻结source-intended stance满足真实sphere-floor contact，同时保留动作语义与Silver其余门。
- 干预：stance distance=0等式、全帧不穿透、swing原clearance不下降；root XY/orientation、上肢、1.46x、175帧/30Hz均冻结。
- 对照：Phase30经Phase36更正后的Bronze，同一auditor复评。

## 结果

- stance |distance| p95/max：`[0.0019413135011553224, 0.004822629975323099]`m；全帧minimum=-0.002418m；penetration fraction=0.097143。
- official contact L/R：`{'left': 0.26285714285714284, 'right': 0.09714285714285714}`；collision↔signed-distance exact：`True`。
- correction root-z max=0.0223m、joint max=0.0864rad；root XY exact=True。
- Silver failed：`['stance_speed_each', 'clearance_p50_each_intended', 'flight']`。

## 结论

固定表示/配置未得到真正Silver；失败门=['stance_speed_each', 'clearance_p50_each_intended', 'flight']，equality=False，nonpenetration=False。这否定本生成器，不否定X2或Any2Any。

## 下一步

按门停止；不扫权重、不改root-z阈值、不跑physics/PPO。
