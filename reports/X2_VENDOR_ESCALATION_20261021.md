# X2反馈发布阻塞：厂商支持/适配请求

## 结论边界
当前会话已复现**不运行Sonic策略也出现的反馈发布阻塞**，暂不能在不改变厂商通信栈的前提下保证串行250Hz接管可靠。不是已证实Sonic模型/SDK版本不匹配；无法指定必须升级的版本。原版no-publish15s750tick通过，策略及250Hz程序不应丢弃。

## 最近可复现刺激与观察
2026-10-21 13:16（Asia/Shanghai），完全吊起，操作员持急停，无custom writer。精确MC pidfd身份绑定，SIGSTOP→SIGCONT2.000389s；45s独立SoC1观察，暂停前反馈新鲜/官方单publisher。命令静默约2.002s；五路反馈最大接收gap2.009–2.030s（torso2.0068s）。最终官方命令及反馈恢复。source IMU13:16:17.858/17.860报告previous Publish2000.685/2005.495ms，poll/read0.006ms。

此前10/20 powered两轮也有2s反馈冻结和同窗Publish阻塞；10/21 08:30官方EM stop/start单独实验，start前反馈已中断。故stop_app与SIGSTOP均不能作为已验证修复。

## 当前身份与读取限制
SoC0 hal_imu/version.json：aimrte v0.8.18，interface v0.9.0.7；这不是完整整机固件兼容矩阵。MC cfg发布leg/waist/arm500Hz、head100Hz（不改）。ROS graph实际BEST_EFFORT；state/IMU publishers TRANSIENT_LOCAL、readers VOLATILE。在同host隔离域的相同QoS reader churn8次未复现2s（publish max1.705ms）；不覆盖厂商wrapper。

当前用户读取/proc/2252/environ与maps遭PermissionError，未绕过/提权；active进程加载的middleware库与XML实际生效情况未完整确认。历史日志libfastrtps2.6.10不能替代当前maps证明。未改DDS XML、QoS、SHM、MC/HAL服务或安装binary。

## 请厂商回答并给验证材料
1. 此aimrte/interface组合HAL Publish对停止/退出的BEST_EFFORT reader是否可能发生SHM端口锁或2s health_check等待？该2000ms等待实际来自哪一层？需源码栈/符号或已知issue，不只“重启”。
2. 当前机型正确官方自定义控制交接方式：如何停止MC command发送，同时避免反馈源阻塞，恢复时如何确认mode与唯一writer？不能把本地enum3当McAction3。
3. 哪个明确整机固件/SDK/aimrte/interface组合修复此问题？提供release note及回退/消息ABI兼容矩阵；确认Sonic-X2 s1ft16000是否受支持，不能凭SDK日期判断。
4. 若建议UDP-only/禁SHM，应提供**厂商进程侧**配置及安全重启步骤、端到端topic发现与延迟验收；仅改custom reader无法证明上游不阻塞。

## 最小后续路径（不自动执行）
先获得上述机制/配置或允许的精确只读诊断权限；隔离副本验证transport变更，维护窗口再经用户批准重启受影响服务/机器人，完成无策略交接反馈连续性与MC恢复门；只有通过后才做串行250Hz有界吊起接管。保留model/PD/保护不变，不试独立writer带载，不强杀writer，不自动清SHM。

## 工件
reports/X2_MC_PAUSE_COVERED_20261021.json、X2_WRITER250_DROPOUT_CHRONOLOGY_20261021.json、X2_DDS_LIFECYCLE_20261021.json及X2_WRITER250_CONTINUE_20261021.md。最近完整原件/tmp/x2-pause-covered-1792559758100954000；SoC0 source rotated /agibot/log/latest/hal_imu/hal_imu.log_2_20261021_131618（应及时保存以免轮转）。没有承诺升级能修复，也没有确认当前版本无法通过厂商适配解决。
