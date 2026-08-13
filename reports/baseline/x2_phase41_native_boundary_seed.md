# X2 Phase41：Phase21 原生稳定边界选择

Phase34 的650个 coherent 1s-safe anchors 中，按“lower15 到 Phase15 首尾均值的L2最小、index最早打破平局”选择 trace index 10（stand 0.0s）。选择过程不读取后续SLSQP结果。

结果：lower15距离0.878rad；root z 0.6522m、tilt 0.0366rad；joint limits无越界；双支撑COM margin +78.9mm。左右脚最低active12 signed distance分别-0.306/-0.306mm，符合原生soft-contact状态。

每脚12个sole sphere的高度不应被误读为全部同时贴地：p95绝对距离约22.38mm。Phase21硬合同采用“active sphere最小signed distance接触、其余sphere不穿透”的真实几何语义。

本阶段1次FK、0 physics、0训练。该状态只解锁Phase21首尾边界，不解锁raw replay或PPO。
