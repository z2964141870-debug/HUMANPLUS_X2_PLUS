# X2 固件升级验证（只读快测）2026-10-22

## 结论
升级后（v1.1.0 线）**稳态反馈流完全健康**：五路反馈全部连续，最大间隔 33ms（阈值 500ms），无任何 ~2000ms 的 Publish 阻塞事件。旧版稳态本就健康，因此本结果**不能**证明旧病根（stop_app/暂停引发的传输瘫痪）已被修复——回归测试见"下一步"。

## 环境变化对照（官方 changelog vs 实测）
| 项目 | 升级前 (v0.9.0-rc7) | 升级后（实测） | changelog 依据 |
|---|---|---|---|
| 内核 (SoC0) | 5.15.148-tegra | **6.1.99-rt36 (PREEMPT_RT)** | 文档未单列内核变更 |
| hal_imu interface | v0.9.0.7 | **v1.0.0.6** | v1.0.0 起"消息定义与 topic/service 变动" |
| aimrte | v0.8.18 | **v0.8.24** | — |
| SoC1 SDK | 作者部署 env | aimdk **v1.0.0-ga424add**（源码态，重新落回 home） | — |
| MC 二进制 | mc_app_main | **aima-mc-app-main** | 非公开接口，changelog 未列 |
| 用户数据 | — | 转移到 `~/.old_agi_home.bak`（两台 SoC 均确认存在） | v0.9.0"特定路径留存"、v1.0.0"目录 ~/.old_agi_home.bak*" |
| JointState ABI | 含温度/电压 | **移除 coil_temp/motor_temp/motor_vol，新增 error_code** → 旧二进制全部不兼容，候选须重编 | v1.0.0 明细 |
| 状态查询 | 无官方接口 | **/aima/mc/common/state (10Hz)** 含输入源/动作/姿态/速度 | v1.1.0 新增 |
| 开发者模式 | 无 | MigrateSystemState/GetSystemState 可用（本次未动用） | v1.0.0 新增 |
| 二开环境 | 直接跑 | v1.1.0 仍可直接跑；**v1.2.0 起强制容器化**（需迁移） | v1.1.0 明细 |

## 25s 只读探针（SoC1，2026-10-22 22:58，BEST_EFFORT 订阅，零发布零写入）
| 流 | 平均 Hz | 最大间隔 | 当前时延 |
|---|---|---|---|
| leg 状态 | 500.7 | 33ms | 0s |
| waist 状态 | 500.6 | 33ms | 0s |
| arm 状态 | 500.6 | 33ms | 0s |
| head 状态 | **334.7**（旧版 100Hz） | 33ms | 0s |
| imu torso | 499.5 | 33ms | 0s |
| imu chest | 498.8 | 33ms | 0s |
| 命令流 ×3 | 0（无人在控，符合吊起+无策略状态） | — | — |

## vendor 日志（开机~22:58）
- hal_imu.log：Publish cost 警告共 16 条，最大 **30ms**（启动瞬态），**无 2000ms 级**。
- mc.log：joint_command "too slow" 警告 2.3–3.3ms（周期 1ms），约 22–101 条/分钟；mc_t2 任务 2.45ms vs 1ms。属于启动段负载特征，需观察是否收敛。

## 未验证（旧病根回归测试，需另行授权/计划）
1. EM stop_app→start_app 循环是否还会引发全网反馈瘫痪（旧版根因一）。
2. covered SIGSTOP/MC 暂停覆盖是否仍使反馈黑屏 2s（旧版根因一的另一表现）。
3. deploy-internal 订阅饥饿是否仍在（旧版根因二）：需要重跑作者 deploy 流程（属接管，需明确授权）。
4. 候选 v02_state_executor 必须按新 ABI 重编后重建硬件门。
