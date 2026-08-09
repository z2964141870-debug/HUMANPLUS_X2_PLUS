# X2 WBT Phase38：hard-constraint feasibility sweet-point

## 裁决

- **PHASE38_LOCAL_HARD_CONSTRAINT_WINDOW_INFEASIBLE_OR_SOLVER_FAILED**
- 只运行source-intent预注册的首个完整DS→SS→DS窗口：frames `[0,25)`，未读held、未运行physics/PPO。
- solver：`scipy.optimize.SLSQP on one preregistered 25-frame nonlinear window`；这不是Phase37 soft-weight扫描。

## 硬约束

- intended stance official sole signed-distance：`0 <= d <= 0.5mm`。
- 全帧nonpenetration；swing clearance不得低于原Phase30；stance speed<=0.10m/s、excursion<=0.03m。
- joint limits/root-z correction bounds、每步qstep<=0.15rad；rootXY/root horizontal acceleration因表示冻结。
- source-fit keypoint仅作objective，不取代接触硬门。

## 结果

- solver success/status：`False` / `9`；message：`Iteration limit reached`。
- iterations/evaluations：`100` / `307`；objective：0.206726。
- max constraint violation：0.0014651。
- stance distance p95/max：`[0.0010559829167341723, 0.0012724414671282007]`m；minimum distance：-0.000249m。
- stance speed max：0.1440m/s；excursion max：0.0189m；qstep max：0.0579rad。

## 结论

当前表示/边界下，真正非线性硬约束局部窗未取得可行证书；不得把Phase37 soft改善晋升为Silver。

## 下一步

按门停止；不换窗口、不放宽阈值、不扫参数、不运行完整轨迹或physics。
