# BASE Phase33：closed exact-state vs Phase28 visible-state matched fork

## 结论

Phase33 得到了决定性但范围明确的结论：**同一 closed episode、同一 MuJoCo、同一未来 ctrl 下，完整 closed 状态能够逐步精确复现未来 50 ms；Phase28 从 93D＋root telemetry 重建的可见状态不能。**

更关键的是，一次被主动判无效的预检显示：如果保留 mmap 的精确 `qpos/qvel`，即使把 time 和 `qacc_warmstart` 清零，再 `mj_forward`，未来 50 ms 仍能近乎逐位复现。因此当前最早分叉的主缺口不是 MuJoCo 版本，也不是 warmstart 单独造成，而是 **ROS telemetry/93D 的采样时刻与状态重建并不等价于 closed 的真实 post-step 物理状态**。

## 假设

Phase28 direct runner 从第一条完整 93D row 重建 `q/qdot/root`，并把它当作新的 MuJoCo 初态。若这份 visible reconstruction 等价于 closed 状态，则在未来输入完全相同的情况下应复现 closed 后缀；若不等价，则分叉应从前几十个 physics tick 内稳定增长。

## 干预

固定 Phase32 同一 episode 的 `t=0.226s` post-step：

- A：恢复捕获到的可用 `mjSTATE_INTEGRATION`，包括 time、精确 qpos/qvel、ctrl、warmstart；
- B：取同一 episode 第一条完整 stand 93D row，严格调用 Phase28 的 `decode_row` 重建 qpos/qvel，再用 fresh `mjData + mj_forward`；
- 两支未来每 1 ms 都使用 closed mmap 已实现的同一 `ctrl`；
- 只看 10/25/50 physics tick，不运行 actor、不重算 PD、不扫字段和阈值。

## 对照与合同纠正

第一次预检错误地把 mmap 精确 qpos/qvel 当成了 B 的“visible state”。它不能回答 Phase28，但提供了一个有用的负证据：精确 qpos/qvel 配合清零 warmstart/time，50 tick 仍能复现，所以 warmstart 单独不是主因。

该预检已显式标记 invalid，没有静默覆盖。有效实验的 B 随后改为同 episode 的真实 93D/root `decode_row`；锚点、未来 ctrl、horizon 和 `1e-12` 判据均未改变。

## 结果

初态即存在可测差异：

- A vs closed：qpos=0、qvel=0、time=0、warmstart=0 error；
- B vs closed：qpos absmax=`2.85e-5`，qvel absmax=`8.60e-3`；
- B 的 time=0、warmstart L2=0；closed warmstart L2=46.50。

未来同 ctrl 的误差：

| Horizon | A qpos / qvel | B qpos absmax | B qvel absmax | B root XYZ L2 max | Contact |
|---:|---:|---:|---:|---:|---|
| 10 ms | 0 / 0 | 1.04e-4 | 1.65e-2 | 8.68e-5 m | 相同 |
| 25 ms | 0 / 0 | 4.09e-4 | 2.17e-2 | 2.15e-4 m | 相同 |
| 50 ms | 0 / 0 | 1.06e-3 | 3.52e-2 | 4.27e-4 m | 中途出现不一致 |

A 的 qpos/qvel 全程精确，contact geom pairs 全程一致；因此 Phase32 observer 捕获到的状态和未来 ctrl 足以跨进程复现 closed 物理后缀。B 的小初始误差则单调放大，并在 50 ms 窗口内改变过接触对。

## 结论

1. **Phase28 的 visible-state replay 合同被证伪。** 93D＋root telemetry 不是可无损恢复的 MuJoCo post-step snapshot。
2. **closed wrapper 的物理并不神秘。** 一旦使用 exact state 与 realized ctrl，direct official MuJoCo 能精确复现。
3. **warmstart 单独不是主因。** 精确 qpos/qvel 的预检在清零 warmstart/time 后仍复现。
4. **剩余缺口属于采样/重建层。** 候选包括 callback 相对 physics 的 26 ms offset、1 ms 内采样点、root velocity、joint q/dq float32 化及 quaternion reconstruction；本阶段没有逐字段消融，不能给它们排序。
5. **这解释的是最早分叉，不自动解释整段失败。** 50 ms 证据足以否定旧 replay 合同，但不能直接声称它是长时间倒地的唯一原因。

## 下一步

停止继续猜 hidden solver state。若继续 BASE，应建立一个“exact-state qualification”工具：所有 direct-vs-closed 物理比较必须从 Phase32 observer 的 post-step snapshot 起步；93D 重建只能用于 policy observation 对照，不能再作为物理 ground-truth reset。

随后最小可证伪问题应变成：在 exact physical state 保持不变时，closed 与 direct 是否对同一 actor/PD command application 时序一致。若一致，则 Phase28 失败应归因旧初态合同；若不一致，再审计 command receipt/application，而不是重新扫 MuJoCo、PD 或 prepare 参数。

## 证据

- 紧凑结果：`reports/official_x2/phase33_matched_hidden_state_fork.json`
- 原始有效结果：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase33_matched_fork/valid_phase28_visible_fork.json`（SHA256 `c883091d...6b69`）
- 无效预检：同目录 `preliminary_invalid_exact_qdq.json`（SHA256 `870c1d35...f9c`）
- 合同纠正：`reports/official_x2/phase33_visible_contract_correction.json`
- 原始预注册及修订：`reports/official_x2/phase33_matched_hidden_state_fork_prereg.json`、`phase33_matched_hidden_state_fork_prereg_amendment.json`
- Tests：3 passed；无训练、无 WBT、无 ROS replay、无真机、无 Git/百度操作。
