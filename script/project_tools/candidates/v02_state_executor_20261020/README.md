# 方案 A：状态接收 executor 隔离候选（2026-10-21）

## 状态与边界

用户批准尝试 A。已验证的ARM候选为state-only隔离加发布快照/终止门，control与writer仍串行；最新源码另有默认关闭、只准模拟话题的independent-writer-offline选项；连接恢复后ARM构建通过，serial 8/8、independent 8/8、refusal 9/9通过。该选项仍只准fake-HAL实验，不能宣称解决真机stale或已可硬件部署。当前隔离候选未向真实HAL发布命令，未停止MC、重启、改固件或覆盖安装binary。

## 更正先前结论

- CONTROL 后五路 stale 是观测事实；executor 饥饿只是可检验假设，不能凭外部订阅器新鲜唯一确定原因。此前 README/提交信息的“confirmed/完全锁定”应撤回。
- 作者实际固件版本未知；SDK 发布时间不能证明作者运行 >=1.1。SDK 归档不是整机升级固件，版本不匹配未被证明。
- 厂商 extra 的 RL_DEFAULT=3 是示例本地 enum，不是官方 MC API。官方 McAction 的 3 是 DAMPING_DEFAULT，**禁止互换**。
- 厂商 RL 示例 README 明确仅仿真验证，不能照搬增益、模型、交接或关机逻辑。独立审查发现示例共享枚举竞争、关机阻尼未保证发送。
- 吊起时 grav_z 稳定、alpha=1 或 tick 数不能证明自主平衡。

## 实现

基线为 writer250 的原模型/观测语义，保留 QoS、250 Hz、50 Hz、500ms/3tick stale 和其阻尼处理。五路 state callback group 设 auto-add=false，仅绑定独立 SingleThreadedExecutor。owned worker 周期 spin_some，既不 detach，也不取消主控制线程。所有 callback owner 构造完成后启动，销毁前 join。spin_some 的时间预算不能抢占阻塞 callback，因此 join 没有硬实时上界。

接收线程异常交由控制拥有的 atomic 通道：STANDBY/INIT/WAIT（尚未写命令）退出；CONTROL 保留原生 stale->damping，不盲目回站姿或直接结束 writer。另修复 GO 跨线程写 plain double 竞争和阻塞 getline 导致的退出 hang。

## 测试证据（阶段性）

- clang++ ASAN+UBSAN 生命周期：200 次立即 start/stop、重复 Stop、销毁后无 callback、异常保存/通知：通过。
- SoC1 ROS Humble 合成测试，domain=173、localhost-only，仅 /plan_a_synthetic/source_[0..4] UInt64，无真实 HAL topics。每个源约500Hz，主线程每次故意阻塞80ms。3s 收到各1450条，控制阻塞期间源0收到1437条，join 后不再回调：通过。这证明该实现隔离有效，不证明原机器人 stale 根因已修好。
- 历史state-only阶段binary SHA256：8c98d63ac493fe3409ddf1eaf3dd7b79fdca1b25588460c62a4c867eb22341fa。最新已验证串行ARM binary为6c47c178abbbc1a0e9662049396a146e287fb5af897a567bebd69958bd514c30；不是当前新增独立writer源码的构建产物。
- test_obs_builder 直接执行七组检查通过（exit 0）。CTest **失败（exit 8）**：原 CMake 用 ament_add_gtest 包裹自带 int main() 的普通测试，执行成功但没有生成 gtest XML；不能写作 CTest 通过。本轮不悄悄修测试框架。
- 初次 colcon 配置因 rosidl_typesupport_c discovery 失败；改用 source ROS 后直接 CMake + explicit aimdk_msgs_DIR 成功，不修改系统依赖。

隔离工作区：/agibot/data/home/agi/x2-v02-state-executor-20261021.rMYM0G
现用 ws/install 与 writer250 原件均未覆盖。

## 尚未通过的安全门

历史串行ARM版本8/8 fake-HAL场景通过，包含逐关节原生阻尼和ramp/HOLD首批完成确认。最终新增独立writer、partial失败锁存、harness discovery/cleanup强化已通过ARM构建与serial8/8、independent13/13、refusal11/11。详见VALIDATION.md，勿把历史结果套用新源码。独立writer控制liveness失效策略和真实MC恢复链仍未通过安全门；fake-HAL测试不能验收真机250Hz或balance。最终binary为2c846ec5379d6172241de6122ef337f2fd326c12462a485fd228836690ae8e12；逐项证据integration_results_final.json/refusal_results_final.json，硬件禁止决定HARDWARE_GATE.md。
后续真机仅在通过上述门与现场安全条件后做有界吊起测试，不自动进入承重/行走。保留此前失败证据，不复用可绕过授权/kill loaded writer 的旧 driver。
