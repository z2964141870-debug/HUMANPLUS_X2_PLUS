# Stage 0：MC stop_app/start_app 隔离测量（2026-10-21 08:30）

执行：08:26 双 SoC 重启后的干净系统上，双端 75s 只读 observer 先行（探针 02eeff…，
预先匹配），随后从 SoC1 以与 deploy_x2.sh 相同机制 POST `stop_app`（08:30:04.025，
回复"应用已停止"）→ 等待 → `start_app`（08:30:10.827，回复"启动应用: mc"）。
全程零自定义进程；observer 未换新、未退出。

## 结果（决定性）

1. **命令静默正确**：stop 后 1 秒内 cmd 降至 0（预期行为）。
2. **反馈在 stop 后 ~4 秒崩塌**：08:30:05-07 state/IMU 仍在流（甚至冲高 727/751 Hz），
   08:30:08-09 双端十路同时归零；此后到观测窗结束（08:30:40+）**始终未恢复**，
   只有间歇性百帧级突发后再次归零。
3. **IMU Publish 阻塞复现**：本 boot 原本只有 8 条开机瞬态；循环后 08:30 分钟 127 条、
   08:31 分钟 101 条、08:32 分钟 123 条，含双通道 **2000.6/2016.6 ms** 阻塞（08:31:28）。
4. **SHM 损坏扩散**：start_app 生成的新 MC 进程（PID 2328 → 7130）在 08:30:42 起对旧端口
   报 `NOT Healthy`；drp / housekeeper / app_proxy / teleop_bridge 同样命中。截至 08:34 已 16 次。
5. **新 MC 收不到前置**：mc.log 尾部持续列出等待的 6 路输入（leg/waist state、chest/torso IMU、retargeting）。
6. 循环后传输**未自愈**（08:34 仍持续 17-19ms 级卡顿 + MC 等输入），与 10-20 22:23 持续故障态一致。

## 结论（根因闭环）

**本固件（v0.9.0-rc7）上 EM `stop_app` 会终止 MC 进程，其 FastDDS SHM 端口残留为损坏态，
阻塞 hal_imu / hal_ethercat 的发布（秒级），并污染后续新建参与者；`start_app` 的新 MC 也无法
收到输入。** 该病理与模型、PD、自定义 writer、observer、no-publish 计算全部无关——
Stage 0 以零自定义代码完整复现了 10-20 四次 powered 接管的反馈空窗 + 持续恶化 + SHM unhealthy。

作者 deploy_x2.sh 的 standby 顺序无法规避：其交接仍走 EM stop_app，反馈在 stop 后 ~4s 崩塌，
无论先建实体与否、无论 500Hz 还是 250Hz binary。

## 本机已被证明可行的替代：SIGSTOP 交接

- 2026-08-28 ledger：SIGSTOP 交接 + 250Hz writer，反馈 73 s 连续新鲜（age 0-3 ms）。
- 2026-10-20 15:02：本工具链再验证 2.005 s 有界 SIGSTOP——命令静默、state/IMU 持续新鲜
  （最大报告年龄 3.4 ms）、恢复验证通过、零 IMU 卡顿、零 SHM 警告。
- SIGSTOP 冻结而非杀死进程：DDS 端点保留、无端口残留、无新进程在污染窗口内创建。

## 修订后的 Stage 1（待执行）

作者顺序 standby + SIGSTOP 静默 + 用户指定的 500Hz 原 binary（失败才换 250Hz 候选）：
watchdog → kplanner → pose gate → `deploy_x2.sh --no-stop-mc`（保留 y/N 门）→ standby 就绪后
以 pidfd 校验的精确工具 SIGSTOP mc_main → 核对命令静默 + 反馈新鲜 → y 触发 CONTROL（吊起 3-5 s）→
Ctrl-C 软停（RAMP_OUT → HOLD_FOR_MC）→ SIGCONT 恢复 MC → 复核官方单 writer 与反馈。
全程 observer 预先在录，接管窗口零新进程。

## 前置条件

当前传输已损坏（本目录证据），**需现场再重启一次机器人（保持吊起）**以获得干净 SHM 域后再执行 Stage 1。

## 文件

| 文件 | 内容 |
| --- | --- |
| `stage0_soc1.jsonl` / `stage0_soc0.jsonl` | 双端 75s observer 原始流（每秒计数见上表分析） |
| `stage0_cycle.log` | stop/start POST 精确时间戳与 EM 回复 |
| `stage0_soc*.stderr` | observer 退出码（exit=2，基线失败，符合反馈崩塌） |
