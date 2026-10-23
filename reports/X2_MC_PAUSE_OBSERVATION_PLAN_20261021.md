# MC短暂停统一观测修订（尚未再次执行）

## 目的
补齐13:11短暂停观察器提前退出的证据缺口。保持无custom writer；不调用EM stop/start，不改SDK/固件/QoS/频率。

## 单一编排要求
1. 在控制主机记录monotonic起点后启动60s订阅observer；时长从启动请求前计预算，不跨host比较monotonic。
2. 有界等待baseline_ready，observer子进程必须仍running；latest sample五路反馈和四command新鲜、sole official graph、custom进程不存在。
3. 紧接signal前调用observation_window_gate.check_window，剩余窗口至少2s暂停+10s恢复+5s余量。不足时退出，不发送signal。不得继续使用15s人工多步采样。
4. 同一MC host本地工具用pidfd绑定boot/PID/start/argv hash。只准2s，finally CONT；该工具无custom writer时可用，不适用于带策略交接。
5. 保存pause实际UTC/elapsed和全程observer。验证暂停时command计数停增、五路反馈持续新收；CONT后四command新流且唯一official、反馈持续>=2s。只graph或CONT返回不是通过。
6. 观察器提前结束/网络不明时不追加MC动作，留原件，检查MC实际状态与恢复；无受载writer可kill。

## 验证
新增纯窗口检查11项unit通过，包括旧15s窗口拒绝、已退出、未ready、时钟回退/NaN/不足恢复余量。它只防采样编排错误，不是硬件安全证明或执行授权。已接入非执行性统一session状态机（pause_observation_session.py），11项fake依赖测试通过：无ready/observer死/launch延迟/身份变/未知pause状态/CONT失败/恢复无流均拒绝，禁止自动重试。尚无live适配器，不伪称真机整体验证完成。

## 当前结果界限
13:11:02–04 MC STOP/CONT2.00027s，同PID后续Sl；CONT后15s反馈/四official command流通过。暂停期间缺采样，不把它算成反馈连续。已有10/20 2s暂停正对照，仍需谨慎区分历史与本轮。

## 下一关
先实现上述single-owner统一编排并离线模拟缺ready、observer中途退出、身份变化、SSH断开、CONT失败；再运行一次覆盖齐全的无策略暂停，最后另批策略接管。用户已说明持续持急停，条件不变不重复问；不要将本次短暂停授权扩大为策略接管授权。
