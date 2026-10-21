# Stage 1 运行记录（2026-10-21 上午）：SIGTERM 交接打通 + CONTROL 阶段部署内订阅饥饿待修

## 执行序列（run1-13，全部吊起，全部有 observer/driver 事件记录）

| run | 静默方式 | binary | 特殊 | CONTROL 结果 |
| --- | --- | --- | --- | --- |
| 1-2 | SIGSTOP（工具 bug 未生效/冻结） | 500Hz | - | 未到 CONTROL |
| 3-4 | SIGSTOP 40s | 500Hz | driver 时钟 bug | 未到 trigger（quiet 门失败） |
| 5-7 | 修复过程 | - | 归属发现延迟/baseline 竞态 | 未到 CONTROL |
| 8 | SIGSTOP 75s | **500Hz 原版** | 全链路走通 | **+1.7s stale**（四路 0.52s） |
| 10 | SIGSTOP 75s | **250Hz 候选** | marker 路径修复 | **+1.5s stale**（五路 0.50s） |
| 11 | **SIGTERM 优雅退出 MC** | 250Hz | ramp 完成 alpha=1.0 | **+3.2s stale**（3s CONTROL） |
| 12 | SIGTERM | 250Hz | intra-op 3 | **+0.7s stale** |
| 13 | SIGTERM | 250Hz | **零 observer+极简 driver（无 rclpy）** | **+0.66s stale** |

## 已确证结论

1. **SIGTERM 优雅交接链全部打通**：MC 对 TERM 响应有序退出（Shutdown Succeeded），
   standby→静默验证→trigger→CONTROL→软停 exit 0→start_app 恢复，全流程多轮可复现。
   EM stop_app 的永久传输毒化（昨日 Stage 0）被彻底绕开；退出窗口仅瞬态 IMU 卡顿。
2. **CONTROL 阶段的 stale 是 deploy 进程内部订阅饥饿**，边界：
   - 与 writer 频率无关（500/250Hz 都 stale）
   - 与 MC 状态无关（冻结/已退出都 stale）
   - 与 intra-op 无关（1/3 都 stale）
   - 与外部监视负载无关（零 observer 也 stale）
   - **外部订阅者（driver/observer rclpy 进程）同期全程 0.02-8ms 连续新鲜**
   - 签名与 2026-08-25 事件一致（writer 正常、状态订阅饿死 → 原生 stale → 纯阻尼）
3. no-publish 计算（10-20 15:30）干净 —— 差异只在 writer 发令。

## 下一步（唯一未试的修复方向）

在候选源码实施 **callback/executor 隔离**（`script/project_tools/candidates/v02_state_executor_20261020/`
预留路线，8 月 D-002 同族问题）：writer 定时器与状态订阅分回调组/独立 executor，
或在 QvhMAD 隔离树重建后重跑 Stage 1。这是代码级修复，不再是参数级。

## 证据

每个 run 的 driver 事件、runtime.log、tick.csv、flight recorder 均在
SoC1 `/agibot/data/home/agi/x2-stage1-20261021/`（sigstop500_run1 / sigstop250_run9 /
sigterm250_run11 / sigterm250_run12 / minimal250_run13）。本目录存关键快照。
