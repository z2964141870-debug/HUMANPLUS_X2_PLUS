# A 第一轮验证结果

- 原 binary 未覆盖：0b02df145874c3e4c6cb05ea4778f09b1bae372212c5bf3754f561b64791864a。
- 新隔离 binary：8c98d63ac493fe3409ddf1eaf3dd7b79fdca1b25588460c62a4c867eb22341fa；路径 /agibot/data/home/agi/x2-v02-state-executor-20261021.rMYM0G/native-build/x2_deploy_onnx_ref。
- ARM 构建完成 exit0。CTest exit8（普通 main 测试被 ament gtest 包裹，没有 XML）；直接运行7组观测检查 all OK / exit0。
- ASAN+UBSAN 200次 worker 生命周期/异常测试 exit0。
- 修订后 ROS 合成复测 exit0：received=1460,1460,1460,1460,1460; received_during_80ms_control_blocks=1448; stopped_before_destroy=1。
- 测试 domain173 + localhost-only，仅自定义合成 UInt64 话题，无真实 HAL 命令或 MC API。
- 未启动候选 deploy、未停止MC、未改固件。实际机器人能否安全接管仍未通过。

独立审查三项阻断已修：worker启动/销毁顺序、可取消stdin reader、接收异常独立控制通道。后两项最终源码已原生编译；尚需实际 AimdkIo 和整进程故障注入验收，不以 fake/UInt64 测试代替。

Git payload 审计脚本虽然 exit0，但本机缺 rg 导致凭据扫描未执行，不能记作完整审计通过；本候选目录另用 grep 工具扫描上述凭据模式零匹配，git diff --check 通过。暂未 commit/push。

## 第二轮完整进程集成

新binary SHA256 d427a8314aaba53971594e7ea36a7e25b799880605832682bed08198e8299ef4（覆盖上述旧候选hash，仅隔离树）。新增每包size验证（消除有效首包后短包越界），支持ROS参数移除/完整topic remap。

真实AimdkIo消息 + 完整s1ft16000 ONNX候选，在domain173/localhost/每进程唯一fake话题运行：healthy、malformed、stdin、stale、late_malformed **5/5通过**，四组命令观察计数均0。正常推理exit0；启动坏包exit4且不进入CONTROL；stdin开启时SIGINT正常退出；断流与CONTROL后截短包均触发原生stale->PURE DAMPING。后者exit4，无SIGABRT。详细路径/断言在integration_results.json。

边界：zero command观察仍缺正对照+discovery barrier，因此是no-publish辅助证据，不宣称可验收writer输出；合成传感器不代表真实动力学。唯一topic remap + ROS域不是OS网络隔离。

独立writer审查发现需要先实现coherent command envelope、四topic整批publication gate + terminal release ack，避免release后多发与挂起推理期间无限重发旧命令。本轮完成未接入deploy的command_publication_gate原型；确定性中途停顿测试验证release等待整批完成、永久不可rearm、ack后1000次调用不发布，ASAN/UBSAN/TSAN通过。**未据此拆writer线程**，控制liveness策略需明确，不能悄悄增加更严格阈值。

## 第三轮：接入完整命令快照与终止发布门

当前binary SHA256：2a6f42d9172319973f850d092b80e5a9966f1276b5e8cda80ad16ad4299da7b9。完整命令提交覆盖原9处ready=true路径；writer从门内不可变快照发完整四话题批次。控制/writer仍串行。正常控制退出路径关闭门（等待批次完成）-> 可选清HOLD sentinel -> ROS shutdown。异常/强制signal不是已验收的正常交接。

独立审查未发现串行接入锁逆序、自锁或漏提交；修正HOLD清理顺序。ARM构建+6项fake-HAL进程矩阵exit0：原5项 + fake_publish正对照。正对照唯一remap模拟接收端各收到728条，exit0且观察到WRITER_RELEASE_ACK；其余各组命令0。详细在integration_results_gate.json。ASAN/UBSAN/TSAN publication gate测试通过。

测试使用domain173、localhost、唯一话题remap，不向真实HAL发布。未用这些结果宣称稳定250Hz、DDS投递队列排空、实际阻尼输出或机器人balance成功。独立writer仍未接入；阻断是控制liveness失效时如何沿用已有安全策略而不无限重发旧命令，需要明确设计/审查与离线注入。不得悄悄新增更严格门限。

## 第四轮：真实阻尼输出与HOLD首批确认

当前binary SHA256 6c47c178abbbc1a0e9662049396a146e287fb5af897a567bebd69958bd514c30。ARM构建及8项fake-HAL测试通过。新增fake_stale检查完整原生Kd向量与Kp=0（head原生Kd=0，不擅改增益）；fake_hold检查ramp结束后首个四话题HOLD批次全部返回，再触碰ready哨兵，exit哨兵后关发布门并删除ready。sentinel open/write/close失败不锁定成功，后续tick重试。no-publish不产生首批确认，也不通知可硬件交接，这是有意的fail-closed行为变化。

command_publication_gate新增mode/enable coherent字段及成功批次generation确认；disabled可重新commit启用，terminal release不可。ASAN/UBSAN/TSAN单元测试通过；尚未把deploy所有mode transition接入该语义，不宣称完整独立writer已实现。

新增test_independent_writer_liveness.cpp，无ROS、无任何网络或硬件端点：控制阻塞80ms时独立state/writer继续进展，release仍能终止发布；ASAN/UBSAN/TSAN通过。同时明确复现旧generation持续重发，这不是安全通过，而是硬件门未通过的证据。没有用native state-stale的500ms推导出未经批准的command TTL。

8项矩阵的fake_stale使用强制隔离signal结束，不应将其release_ack标记true；新版harness删除该项断言而明确signal_sent（正常fake_publish/fake_hold仍要求真正WRITER_RELEASE_ACK）。原integration_results_hold.json保存上一版阶段证据，保留不覆盖。

## 第五轮：硬隔离门控的独立writer源码

新增--independent-writer-offline（默认false）及--offline-fake-prefix。X2Deploy构造前严格核验domain173/localhost1、ASCII nonce前缀、9个ROS最终解析话题与prefix+canonical suffix完全一致，包含IMU override。强制standstill、MC poll0、禁ZMQ/teleop。writer单独auto-add=false group/executor/owned worker；默认保持串行。OnWriter不再拆读state/ready，只依赖不可变no_publish及完整envelope；首个HOLD成功批次确认仍保留。mode作为诊断不等于已经接入独立控制liveness策略。

独立审查核对endpoint覆盖、9条commit路径、worker生存期；发现partial publish unwind与release ACK的竞争。已在gate内部同一锁下先锁存failed/released再抛异常，Release返回失败时禁止正常ACK/清哨兵。新增disable并发/rearm、四个partial失败位置、release与失败批次竞争回归，TSAN及ASAN/UBSAN通过。

**新增ARM构建/双模式8+8项矩阵/9项拒绝测试尚未执行成功。** 上传第一步scp connection closed exit255，后续native build没有启动。旧socket及新连接均SSH banner timeout，22041转发监听存在但不能证明远端连通。未重启机器人/MC/网关，无新增硬件写入。上一已验证ARM binary仍是6c47c178...，不可冒充当前源码构建结果。

准备的harness已支持serial/independent/refusal三模式、唯一UUID前缀、binary/model/params hashes与接收间隔记录；当前未跑过的结果不能算通过。独立控制卡死后重发旧命令仍是明确硬件安全门，不引入未经批准TTL。

## 第六轮：连接诊断及本地隔离验证

PC2/PC1两个转发口SSH banner均超时；Windows bridge22端口可答握手但BatchMode无可用认证（permission denied），旧隧道TCP仍ESTABLISHED。未擅自重建有状态隧道/重启远端。新增ARM编译仍未开始，不能归咎代码构建失败。

run_local_safety_tests.sh可复现本地worker/gate/liveness的ASAN+UBSAN/TSAN全套通过；从实际deploy guard摘取C++主体、mock resolver编译的27项允许/拒绝决定通过。该mock不验证ROS remap API，不能代替目标ARM构建。

harness资源收购移入外层try，嵌套finally覆盖失败清理；所有case必须观察到五state和四command端点匹配，避免未发现订阅也把0命令算成功。此强化版尚未跑ROS。新增实际stop_child抽取测试在Mac遇killpg PermissionError（OS进程组权限限制）exit1，连测试finally同样被拒绝；未换方法绕过。所启动测试child仅本地sleep60，未联网，不是robot writer。这不是cleanup runtime通过证据；仍需Linux隔离域验证。

## 第七轮：持续连接阻断及可复现检查点

连续第5、6、7轮同一PC2 SSH banner timeout（exit255），因此无法上传/编译/验证最新独立writer。未以重复重试代替进展。27项实际源码guard mock测试和worker/gate/liveness全套sanitizer再次通过；实际cleanup函数六项OS模型测试通过（INT退出、TERM退出、KILL退出、group消失、权限拒绝、不可回收超时），明确不作为Linux进程组实测通过。

当前任务应标记外部连接blocked，不是完成。恢复条件：Windows桥接机能再次连通PC2 10.0.1.41:22，或提供可用SSH路径；随后继续隔离树native build和serial/independent/refusal矩阵。无需也未经授权重启MC/固件/HAL。

## 连接恢复后的ARM验证（12:38）

原转发已恢复可达；新建ControlMaster使用既有HostKeyAlias成功认证（未禁用host key验证）。最新guarded independent源码native CMake ARM构建通过，binary SHA256 1058177f6f20108822ab7261415dc26044d06c28f7caf1ecc182929d67391175。

强化harness首次8项失败来自Humble Subscription无get_publisher_count方法，不是候选运行失败；改为node.count_publishers(unique topic)，保留正discovery要求。随后serial 8/8、independent 8/8、refusal 9/9全部通过，整体job exit0。拒绝包括wrong_domain/localhost_off/missing IMU或command remap/IMU override escape/MC service/prefix/global remap/ZMQ。Linux实际cleanup回归（正常/已退出/忽略INT TERM child）通过exit0。

结果见integration_results_independent.json，每项均有5state及4command discovery覆盖。正对照serial各组681–682条，独立722–723条；接收端平均间隔serial约4.41ms，独立4.14–4.15ms，最大约13.7ms/11.2ms。只是本次短测的接收时间，不证明硬实时250Hz或无DDS排队/丢包。

仍不等于真机stale根因修复，不得解除offline-only门。独立control卡死时旧command持续重发、真实MC恢复/唯一writer交接、异常部分批次硬件故障策略仍未验收。## 最终隔离故障验收

最终ARM binary SHA256 2c846ec5379d6172241de6122ef337f2fd326c12462a485fd228836690ae8e12。fault hooks默认0/-1关闭，非default必须过offline门。串行8/8、独立13/13、拒绝11/11通过，整体job exit0。block300ms期间四组writer继续73/73/72/72条，这证明调度隔离，也明确暴露硬件control liveness缺口。四组partial注入均故障exit5而非SIGABRT，无虚假release/HOLD ACK。最终独立静态审查未发现新增可行动缺陷。

逐项证据integration_results_final.json/refusal_results_final.json。现用安装binary原hash未变。HARDWARE_GATE.md决定：**不放行真机，不解除offline-only门**。本阶段隔离候选、源码审查、构建、离线测试及安全门结论已交付；真实机stale与自主balance未证明解决，控制活性策略/真实MC交接属于下一阶段。
