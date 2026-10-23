# 方案 A 离线结论与硬件安全门

## 已完成的隔离实验

- 状态接收独立executor；独立writer仅通过默认关闭的offline选项启用；控制保持50Hz，writer timer保持4ms。完整命令快照、整批publication gate、terminal release与HOLD首批确认。
- ARM native build通过。最终binary SHA256：2c846ec5379d6172241de6122ef337f2fd326c12462a485fd228836690ae8e12。
- 串行8/8、独立13/13、拒绝11/11通过。最终逐项结果见integration_results_final.json与refusal_results_final.json。
- 独立审查、ASAN/UBSAN/TSAN、Linux实际子进程清理通过。已有CTest XML框架失败仍未冒称通过。
- 现用install binary SHA256仍为0b02df145874c3e4c6cb05ea4778f09b1bae372212c5bf3754f561b64791864a，未覆盖。无MC/HAL/固件更改或真实HAL命令发布。

## 明确未通过：硬件部署禁止

1. **控制活性**：完整候选CONTROL故意阻塞300ms时，writer仍发出约72–73条旧command，状态接收仍可保持新鲜。证明调度隔离，不证明安全。无限控制阻塞可无限重发旧命令；没有批准的command-heartbeat期限、分mode故障动作、独立停止路径及late inference不可重启故障的仲裁。不能把原生state 500ms/3 control ticks擅自改成command TTL。
2. **硬件故障处理**：partial publish实测四个位置均exit5，无虚假release/HOLD ACK；这是模拟域终止策略，不是带载退出/阻尼/MC恢复授权。默认异常路径仍有历史风险。
3. **所有权/MC恢复**：首批ACK只是本地四topic调用完成，不是DDS排队清空、接收端确认或MC实际持力。唯一writer/MC PASSIVE→JOINT→STAND交接未实测验收。
4. **时序/快照**：250Hz仅timer意图；接收间隔有抖动/丢帧。五topic非同步源时间，snapshot与freshness各自锁定，不保证同一个状态版本。真机CONTROL后stale根因尚未被证明。

## 下一阶段必须先设计审查

保留offline-only硬门。先确定控制心跳完成语义、deadline依据与分mode故障处理，不暗改原生阈值；补故障仲裁/恢复策略和真实所有权证据后再评审有界吊起实验。不得自动转承重或行走，不可强杀带载writer，不以本报告批准固件/MC/HAL重启。

本阶段交付是隔离候选及离线证据，不是可真机部署版本。
