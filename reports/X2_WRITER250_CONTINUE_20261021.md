# 串行250Hz继续定位（2026-10-21 12:55 CST）

## 基线保留

PC2 uptime约20min；未发现运行中的custom deploy（pgrep仅返回本次查询shell）。原writer250-resume.QvhMAD候选仍在，SHA256 2f1590369e4574c9e4debe89d8f34df19dbfbc21b93627b676e2038106c26ffa。未启动控制、暂停MC、改DDS或发真实HAL命令。

## 本轮现场只读采样

12:53:02–12:53:17（CST）subscription-only 15s，exit0，baseline_ready。远端原件/agibot/data/home/agi/x2-readonly-current.h5hz86/baseline.jsonl、baseline.stderr。observer位于SoC1，BEST_EFFORT depth1，结果是收到的流，不是源实际生产频率。

最终窗口state leg/waist/arm/head/torso IMU约471/442/417/259/299Hz；全程最大接收间隔8.49/8.67/9.34/10.40/10.37ms，最后age均<4ms。官方command也在流动。说明当前空闲基线健康，不说明接管时健康，更不证明ownership。

## 原失败复核：纠正诊断方向

resume_powered3_supervisor.jsonl第7–14行五路计数冻结537/490/464/245/263，而command持续增加；settle第9–16行冻结2535/2312/2214/1024/1607。按mono-age推定最后接收约trigger+0.08s/+0.43s；+2.261s/+2.511s观察到恢复。两个runtime中control持续约20ms，原生stale触发于CONTROL后0.560s/0.940s。

这不支持“仅部署callback饥饿”作为完整原因。厂商日志settle_hal_libraries_logs.txt第18–23行直接记录IMU Publish约2000.659/2007.163ms阻塞（poll/read约0.01ms），但这些是22:40后续现场，不能当作精确初始因果。DDS reader压力/QoS/entity lifecycle、官方交接仍是待检假设；不能据此修改QoS、频率或保护。

## 可复现时间线结果

analyze_writer250_dropout.py计算并断言两轮均有8个冻结snapshot、>1.8s无反馈、command增量>400。每轮五路最后receipt在2.25ms/1.88ms内集中停止，native fatal距独立observer最后receipt约0.539–0.541s/0.555–0.556s。与部署自己的native stale报告一致，不是仅应用内部时间戳异常。wall-minus-mono在每轮变化仅0.0017/0.0026ms（浮点精度有限），仅可用于同host对齐。

缺少初始trigger精确厂商Publish trace，明确不推断唯一DDS机制。结果X2_WRITER250_DROPOUT_CHRONOLOGY_20261021.json。脚本断言通过exit0。

## 原版no-publish对照已完成（12:58 CST）

保持原binary SHA 2f1590...、模型及历史参数，仅max-duration=15并加绝对no-publish，删除历史触发sentinel，MC未切换。实际source OnWriter第3759行在任何publish之前return，独立observer覆盖启动前三秒、15秒推理和退出后窗口。远端原件/agibot/data/home/agi/x2-serial250-nopublish.u0z09qx3/{argv.json,result.json,runtime.log,observer.jsonl,observer.stderr,csv/}。

模型加载→CONTROL 750 ticks/15秒→正常exit0，未出现native stale；observer30秒exit0，五路feedback最大接收间隔leg18.70/waist16.58/arm14.53/head15.24/torso19.90ms。原版推理、订阅加入和退出在本次窗口未复现dropout，不能排除偶发或接管/发命令特有机制。未做GID实际command归属审计，command_sending=false是no-publish配置及源码保证，不应声称本轮独立GID证明零发送。

首次尝试因嵌套shell提前展开PYTHONPATH导致observer缺rclpy，deploy尚未启动即拒绝；失败原件7gmd9gsc保留。改上传独立shell正确source环境后成功。直接help首次缺ONNX库也只是环境失败，没启动控制。

结论：旧版计算链可正常运行，不能说250Hz不可用；下一步应审查反馈源Publish、官方交接与发布端点/QoS，而不是重调PD/model或替换全新writer。真实接管前仍需安全MC恢复及现场条件。

## 补回实际触发窗口源日志（13:00 CST）

PC1旧boot目录原始rotated IMU日志仍存在，本轮只读取回：22:13:04.038471报告torso previous Publish=2000.677ms，chest 04.088722=2052.421ms；第二次22:23:17.624249 torso=1999.859ms、chest17.624017=2000.616ms。poll/read均约0.004–0.059ms。粗估Publish开始约22:13:02.038 /22:23:15.624，和SoC1推定最后receipt（02.039/15.623）相近，**跨host偏差未知且previous-call边界未直接trace，不报微秒因果**。源Publish阻塞与实际失败窗口关联已比22:40 aftermath证据更强，但具体DDS/传输/锁竞争触发仍未知。

紧凑原文保存trigger_imu_publish_recovered_20261021.txt，完整本地/tmp/x2-vendor-imu-trigger.log与/tmp/x2-vendor-imu-settle.log；远端原件未变。此前“初始vendor trace缺失”只描述当时掌握证据，现由本节取代，不删除历史结果。

当前no-publish窗口graph中五路反馈publisher及所有观察到的reader都BEST_EFFORT；不能继续把“自定义RELIABLE reader导致ACK阻塞”当成已成立解释。未修改QoS/持久性/SHM。当前12:58时段IMU日志只见约19–20ms Publish（此次取回文件范围），与observer最大20ms一致。

下一最小实验应先隔离传输/端点生命周期并确认厂商Publish背后的等待机制，而非改策略推理。具体提案：不动真实HAL/MC，在隔离域复现相同BEST_EFFORT+TRANSIENT_LOCAL发布/volatile reader加入退出；真机交接需先修审可验证恢复链，不能直接复用旧timeout带载终止driver。

## 隔离QoS生命周期对照

SoC1 domain173 localhost-only，唯一/plan_a_lifecycle_UUID Imu，BEST_EFFORT+TRANSIENT_LOCAL publisher / BEST_EFFORT+VOLATILE reader。8次独立reader加入/正常退出均成功，5842次publish平均0.124ms、最大1.705ms；未复现2s publish阻塞。loop最大296.8ms包含同步communicate及创建进程开销，不能当作publish阻塞。只是同host Python/rmw surrogate，未覆盖厂商C++ wrapper、跨host SHM配置、MC交接；不据此排除真实DDS机制。结果X2_DDS_LIFECYCLE_20261021.json。

## 下次有界实验审查草案（未授权，不执行）

假设：MC交接/真实命令发布触发源Publish阻塞，单纯推理或reader加入退出在目前对照不够复现。沿serial250原model/PD/rate/原生保护，不启independent writer。先完成双host时钟对齐记录与官方ownership核验，在PC1同步收集IMU Publish成本、HAL状态；PC2同窗记录5路receipt、command GID与CONTROL时间。两阶段分开：先官方交接且无自定义写入的短MC stop/start对照（本身可能掉力/运动，需完全吊起和单独授权），通过恢复门才另批短接管，不一次混做。

任何阶段前需用户确认本轮已完全吊起且有人持急停，明确批准MC切换；预先审查官方恢复方案，确认唯一official command publisher+实际收流，禁止复用旧timeout带载kill或以start_app success认定恢复。运行中feedback blackout保留原生阻尼，不放宽阈值；禁止自动承重/行走。当前只读授权不足以执行该草案。

## 恢复链审查补充（等待现场授权）

旧x2_v02_suspended_probe.py第164/203行timeout+stop_child可强杀writer，不可复用带载；恢复段只有限窗口graph与fresh判断，不充分证明实际唯一official command归属。新增纯函数mc_recovery_evidence.py只用于对照判据：四组唯一official endpoint、新command计数、五路feedback及command接收年龄、>=2s观察窗口。11项unit通过；不是实际MC恢复证明，不包含GID绑定或实际mode/力矩，不自动调用EM。

等待用户明确本次吊起/急停及MC短切换授权。自动goal事件不是现场确认，不执行stop/start。授权后仍先把纯判据与连续成功窗口、GID实收检查接入新无自定义writer的诊断driver，并离线mock超时/取消/HTTP回复丢失等恢复路径，不直接运行旧powered driver。

## 重要纠正：已有官方-only cycle证据，不应重复切换

本轮重新读到logs/x2_stage0_mccycle_20261021/README.md及原始cycle/双端jsonl。08:30:04.025 stop_app，08:30:10.827 start_app；SoC1 08:30:07.428反馈仍新鲜，08:30:08.438五路冻结、age约0.82s，09.448约1.83s，即**start_app前已经故障**。双端独立接收日志一致支持反馈中断，不能归因于策略推理/自定义writer；零自定义控制器是该记录的实验provenance，本轮没有重新执行。之前未充分检索便提议重复官方-only循环是失误，现撤回。

该README用“根因闭环/SHM端口残留”语言过强：原始反馈中断+vendor Publish阻塞+SHM unhealthy支持官方stop路径相关传输故障，但具体内部锁/SHM回收因果未直接trace。本轮不把机制当作证明，不批准SHM清理或固件更改。

最小替代候选：**不调用EM stop_app杀MC，保留端点，通过身份校验pidfd有界SIGSTOP/SIGCONT静默**。已有20261020 15:02约2.005s暂停正对照可支持短暂停，不等于策略接管或长时MC恢复验证。必须先解决“custom loaded writer退出与MC继续”的可靠交接，不能自动到期SIGCONT撞上仍在写的custom writer。

因此无需以SDK升级为前置结论；先审查旧串行250Hz+保留MC进程交接，模型/增益/频率不变。仍需明确现场吊起/急停与暂停MC授权。当前不执行任何MC动作。

## 当前暂停门（13:08后）

用户已明确机器人吊起，尚未明确有人持物理急停及同意本轮MC暂停/恢复。第7–9轮未获得这两项回复；自动goal事件不是现场授权。只读预检MC PID2252在运行，两host无custom deploy（pgrep仅查询shell）。不执行任何signal/EM动作。

暂停工具的finally无条件SIGCONT只适合无custom writer实验，不能直接用于策略交接。最小后续候选仍是serial250+terminal gate，不启independent writer；超时不能单独视为release，ACK不是DDS drain，需正常退出及实际official GID流验证。暂停工具9项、恢复证据11项、抽象handoff contract7项（穷举32组合）通过；非真机恢复证明。

## 13:11 授权短暂停结果与证据限制

用户确认完全吊起、学弟持续持急停、批准本轮约2s无策略MC暂停。条件未变不重复问。MC精确身份PID2252/start1753/boot bec97533-64c6-4a96-892a-6936c0e962b1/argv hash b7f425e9...。13:11:02.506确认STOP，04.503 CONT，stop_to_cont=2.000268667s，工具exit0；瞬时state D仅代表不再stopped，后续ps为Sl同PID。未启动custom writer或调用EM。

**采样编排缺陷**：预先启动observer只有15秒，在实际STOP前已退出（job结束13:10:43，STOP13:11:02），故本轮不能声称暂停期间反馈连续；不掩盖、不自动重复MC动作。初次身份查询shell引号错误，只读失败，已修正。

补做CONT后15秒只读恢复成功exit0，baseline_ready；每组唯一mc_ros2_node publisher，command各6880/6514/5724/1486条，反馈五路最大gap8.14/8.95/9.20/11.27/10.22ms。表明采样窗口官方命令和反馈已恢复，不证明暂停期间或物理持力。远端pause1311-recovery.jsonl、pause1310-observer.jsonl均在隔离candidate工作区；本地/tmp同名recovery留存。

后续应将observer+pause放入同一编排并检查活跃和剩余时长，覆盖暂停及恢复；不复用独立15s窗口人工多步调用。保留原版，不自动策略接管。

## 13:16 全覆盖暂停：候选被否证，停止接管

单一编排45s observer，ready+官方单publisher+feedback新鲜+无custom+足够剩余窗口后执行同PID2252暂停，STOP→CONT2.000389s，整体exit0仅表示工具/采样完成，不代表实验通过。13:16:15.415 command_leg固定1475，16.424仍1475；IMU age从15.415的1.1ms到16.424的0.569s、17.424的1.569s。全窗state最大gap2.009–2.030s、torso2.0068s，**反馈连续检查失败**，结束时反馈和四command恢复。

厂商rotated hal_imu.log_2_20261021_131618第5972/5975行13:16:17.858/17.860记录Publish2000.685/2005.495ms，poll/read均0.006ms。即本轮SIGSTOP保留MC进程也出现源Publish阻塞；撤回将其视为足够修复的假设。历史2s正对照不能替代本轮。无custom策略运行，不可归因于模型/250Hz writer。

完整原件/tmp/x2-pause-covered-1792559758100954000/{observer.jsonl,pause.jsonl,observer.stderr,pause.stderr}；结构化reports/X2_MC_PAUSE_COVERED_20261021.json。该结果不应字段名checks_passed（没有设置），需按反馈gap判为FAIL。当前不再试策略接管、不升级、不清SHM。下一步需要厂商DDS/SHM等待机制分析或受控传输适配；尚不能断言SDK版本不匹配或给出确定升级版本。

## 下一步（最小路径）

1. 从已有原始时间线计算每组最后receipt和CONTROL/stale精确关系，查找厂商初始触发窗口，缺失明确标记；不继续为未经证实原因增加控制复杂度。
2. 必要时对当前官方状态做更长只读对照、读取相同时段厂商日志，确认Publish阻塞是否在未接管时也存在。
3. 选择最小serial候选的no-publish对照，保留模型/50Hz推理/250Hz writer/原生保护。真实handoff另作有界方案审查，继续实验授权不视为可自动暂停MC或带载测试；现场吊起与可靠MC恢复是前置条件。

独立writer只保留offline候选；不以本报告替换旧版或批准承重行走。
