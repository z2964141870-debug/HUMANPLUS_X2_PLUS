# 原生部署饥饿病 A/B 定位（writer 频率扫描）2026-10-23 00:0x

## 结论
**高频报错被定位并绕过：饥饿与 deploy 的 writer 发布频率直接相关。** 用作者源码自带的 --writer-hz 旋钮（重编补齐，默认行为不变）扫描：

| writer Hz | run | CONTROL 结果 |
|---|---|---|
| 500（出厂默认） | run2, run3 | ✗ FATAL（strike 3/3 → E-STOP PURE DAMPING），2/2 |
| 250 | run5 | ✗ FATAL，签名相同 |
| 150 | run8 | ✗ FATAL，签名相同 |
| 100 | run7, run9 | ✓ **通过**（0 strike，完整 STANDBY→CONTROL→软停→exit 0），2/2 |
| 50 | run6 | ✓ 通过，1/1 |

**存活边界在 100–150Hz 之间；100Hz 两连过。** 频繁报错（state stale >500ms ×3 → E-STOP）在 ≥150Hz 必现、≤100Hz 不现——发布负载把进程内状态接收饿死的机制假设成立。

## 变更清单
- ws/src/x2/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref.cpp：4 处最小补丁（CLI 字段 writer_hz=500、help、--writer-hz 解析、writer 定时器周期 1000/clamp(hz,50,1000)）。默认 500Hz 行为与原二进制一致（run2/3 仍 FATAL 证明无行为漂移）。
- 重编 Release + aimdk_msgs_DIR=aimdk_msgs_pc2/share/.../cmake，1min19s，新 sha256 466a9864e01d6b522e87d199ff87c6e1b98e8e07c1ba845703403c7d1167e97d（旧 0b02df14… 备份在 .old_agi_home.bak 可回滚）。
- 驱动器 x2_stage1_sigstop_takeover.py：新增 X2_WRITER_HZ / X2_EXTRA_ARGS 环境变量透传（默认不传时行为不变）。

## 100Hz 运行质量观察（run7/run9）
- C++ 侧整个 CONTROL 窗口状态全程新鲜，无任何 strike；
- 4s CONTROL + 2s ramp-in/out 全链 exit 0，软停 sentinels 正常；
- 机器人吊起+急停持有，MC 45s 有界暂停 5 次全部 auto-CONT 恢复正常。

## 下一步
1. 100Hz 下带 VLA pose 输入的行走质量评估（真实操作点）——确认 100Hz 命令流对 hal 的可接受性；
2. 若行走需要 >150Hz，改走候选方案（独立 state 执行器线程），重编后同法 A/B；
3. 向作者报 bug：写作者源码支持 --writer-hz 但发布二进制缺该旗子，且默认 500Hz 在 X2 上自饿。

## 证据
- logs/x2_native_deploy_20261023/run{3,6,7,8,9}*_runtime.log + *_driver.jsonl
- SoC1: ~/x2-native-redeploy-20261022/run{3,5,6,7,8,9}*/（csv BLACK BOX）

## 官方文档修复结题（writer 频率扫描后续，09:00-09:16）

### 用户指令
"根据官方的文档修修看，我还是感觉100hz不靠谱" → 按官方文档修复根因，恢复 500Hz 契约。

### 官方参考实现（用户下载包 + SoC1 SDK 源）
- 包: aimdk-extra-x2_rl_deploy-v1.1.zip（~/Downloads），同内容也在 SoC1 /agibot/data/home/agi/aimdk/extra/x2_rl_deploy/
- 架构: 订阅回调只锁存(DataBoard 锁+拷贝) → controlLoop(2ms)/publishLoop(2ms=500Hz) 各自独立 std::thread；rclcpp::spin 单线程只服务回调
- QoS: rclcpp::SensorDataQoS()（我们已一致）
- RMW: README 依赖 ros-humble-rmw-cyclonedds-cpp；start_sim.sh export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
- **注意: README 自述"示例程序只在仿真环境中验证通过，控制真实 X2 需重新参数适配"**

### 实验矩阵（今天，机器人重启后 boot_id=7638c43e…，MC pid=3118 start_ticks=2099）
| run | 配置 | 结果 |
|---|---|---|
| 10b | SoC1 CycloneDDS(自制 cdds_prefix 1.3.5+iceoryx) 500Hz 真发布 | FATAL：CONTROL 后 1.24s 断流（比 FastDDS 的瞬时断慢）→ 且 **driver 同步断流 2s**（state_leg 计数 29511 冻结）|
| 11c | 同上 + --no-publish | PASS 0 strikes → TX 是触发变量 |
| 12 | SoC0 同机（FastDDS 默认 profile） | driver 无法发现 DDS 实体（未带厂商 profile）|
| 13 | SoC1 FastDDS + **厂商 ros_dds_configuration.xml（20MB 缓冲+SHM+白名单 10.0.1.41）** + --no-publish | PASS |
| **14** | 同 13 + **500Hz 真发布** | **PASS 0 strikes** |
| **15** | 复验 14 | **PASS 0 strikes（2/2）** |

### 根因（证据链闭合）
厂商所有进程通过 /agibot/software/entry/cfg/basic_env.sh 加载 ros_dds_configuration.xml：
UDPv4 收发缓冲 20MB + SHM(segment 20MB, queue 10240) + 接口白名单。
我们的 deploy/driver 一直用 rclcpp 默认（内核默认 ~200KB 缓冲）→
500Hz×4 命令流(2000msg/s) + 5000msg/s 状态流的组合把接收队列打爆 →
状态断流 0.5s×3 → watchdog E-STOP。100-150Hz"边界"= 默认缓冲的饱和阈值。
100Hz 从来都是症状压制，不是修复。用户的直觉正确。

### 修复（一行环境变量，无代码改动）
FASTRTPS_DEFAULT_PROFILES_FILE=<SoC1 版 ros_dds_configuration.xml（IP 白名单改 10.0.1.41）>
已落盘 SoC1:~/dds/ros_dds_soc1.xml。二进制零改动（--writer-hz 补丁保留，默认 500 不变）。

### 遗留
- SoC0 运行需 source /agibot/software/entry/cfg/basic_env.sh（run12 未带 profile 失败，未重试——SoC1 修复已足够）
- 官方参考架构（订阅锁存+独立线程）可作为后续加固项，当前证据表明非必需
- 昨晚 100Hz workaround 正式退役
