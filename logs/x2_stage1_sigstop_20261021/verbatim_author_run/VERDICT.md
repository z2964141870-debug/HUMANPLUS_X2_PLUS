# Verbatim author run (09:56-09:59)
start_x2_deploy_ritual.sh 原件 + PC2_HOST=10.0.1.41，作者全参数自动点火。
结果：CONTROL 4 秒（tick 200 / alpha 1.0 / grav_z -0.96~-0.99，策略在平衡）后
五路 0.51s 同时 stale -> E-STOP 纯阻尼。与 run8-13 的重构版失败完全同签名。
结论：非重构问题、非交接方法问题——deploy 在本机 writer 激活后订阅饥饿（8-25 事件同族）。
