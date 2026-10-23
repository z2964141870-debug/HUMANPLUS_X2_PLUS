# Sonic-X2 作者咨询邮件草稿（2026-10-21）

## 联系渠道结论

- 作者：**Sitarama Raju Chekuri**（GitHub: [meetsitaram](https://github.com/meetsitaram)，San Francisco）
- **未找到公开邮箱**：GitHub API email 字段为 null（未公开），无 blog/twitter；项目页无联系方式；HuggingFace（tinkerbuggy）抓取失败，历史抓取中也无邮箱。
- **推荐渠道**：在 [github.com/meetsitaram/GR00T-WholeBodyControl-X2](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2) 开 issue（公开、可引用、作者最可能回复），或 HF tinkerbuggy 模型页 discussion。
- 以下草稿可直接用于 issue 正文或邮件。

---

## 英文版（推荐直接使用）

**Subject:** X2 firmware / SDK stack used for Sonic-X2 v0.2 real-robot runs (feedback blackout question)

Hi Sitarama,

I'm deploying your Sonic-X2 v0.2 (s1ft16000, ship-frozen-core c0036a4) on our AgiBot X2 Ultra and hit a reproducible feedback blackout during the MC handoff. I could not find your email, so posting here / writing directly — happy to move wherever you prefer.

Could you share the exact robot-side stack you used for your real-robot experiments (e.g., the ~18 min rough-terrain walk reported 2026-09-12)?

1. X2 variant and **robot firmware version** (our system reports vendor v0.9.0-rc7; hal_imu reports aimrte v0.8.18 / interface v0.9.0.7). Did you (or AgiBot) upgrade the firmware for this project, and to which version?
2. **AimDK SDK / aimdk_msgs version** you built against, and whether it matched the firmware's message ABI (JointState without temperature/voltage, etc.).
3. **MC handoff method** in your deploy_x2.sh flow on the real robot: does your stop step call the EM `stop_app` API, or something else (SIGSTOP/SIGTERM, or the newer Develop_MC mode)? Did you ever see, after stopping MC:
   - joint/IMU feedback stop for ~2 s network-wide,
   - hal_imu logs like "previous Publish() cost: ~2000 ms",
   - FastDDS SHM "NOT Healthy" warnings from restarted processes?
4. Any **DDS/transport configuration** changes (Fast DDS XML, SHM disabled, UDP-only, QoS) you applied on the robot side?

On our unit, a zero-custom-code test (observer-only + EM stop_app → start_app) reproduces feedback collapse ~4 s after stop_app plus 2 s Publish() blocking and persistent SHM port damage, so I suspect a firmware/transport-version-dependent issue rather than the policy. Any pointer to the versions you ran would tell me whether upgrading is the right move.

Thanks for the great work — the frozen-core release and docs are excellent.

Best regards,
<你的署名>

---

## 中文对照（供你确认内容，不建议原文发送）

主题：Sonic-X2 v0.2 真机实验所用的 X2 固件 / SDK 版本咨询（反馈中断问题）

你好，

我们在 AgiBot X2 Ultra 上部署你的 Sonic-X2 v0.2（s1ft16000，ship-frozen-core c0036a4），在 MC 交接时遇到可复现的反馈中断。想向你确认你真机实验（如 2026-09-12 报告的约 18 分钟粗糙地形行走）所用环境：

1. X2 型号与**整机固件版本**（我们系统为 v0.9.0-rc7；hal_imu 报 aimrte v0.8.18 / interface v0.9.0.7）。你或智元是否为本项目升级过固件？升到哪个版本？
2. 所用 **AimDK SDK / aimdk_msgs 版本**，是否与固件消息 ABI 匹配（JointState 已无温度/电压字段等）？
3. 真机 deploy_x2.sh 流程中 **MC 交接方式**：stop 步骤调用的是 EM `stop_app`，还是其他方式（SIGSTOP/SIGTERM 或新的 Develop_MC 模式）？停 MC 后是否见过：全网关节/IMU 反馈中断约 2 秒；hal_imu 日志 "previous Publish() cost: ~2000 ms"；重启进程报 FastDDS SHM "NOT Healthy"？
4. 机器人侧是否改过 **DDS/传输配置**（Fast DDS XML、禁 SHM、仅 UDP、QoS）？

在我们机器上，零自定义代码测试（仅 observer + EM stop/start）即可复现 stop 后约 4 秒反馈崩塌、2 秒 Publish 阻塞及 SHM 端口持续损坏，因此怀疑是与固件/传输版本相关的厂商问题而非策略问题。你运行的版本信息能帮我们判断升级是否正确方向。

谢谢你的出色工作——frozen-core 发布和文档非常好。

## 发送前注意

- 我方证据摘要（可按需附上）：stage0 零自定义复现记录、厂商 hal_imu Publish 2000.685/2005.495ms 日志行、13:16 全覆盖暂停 FAIL 判定——见 reports/X2_VENDOR_ESCALATION_20261021.md。
- 不要在公开 issue 中贴内网 IP、账号、socket 路径等敏感信息；引用日志行即可。
- 若作者回复索要更多数据，先脱敏再发。