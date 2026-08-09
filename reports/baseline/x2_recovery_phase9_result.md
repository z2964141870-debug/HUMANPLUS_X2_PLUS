# BASE Phase9：stateful recovery 5-update paired smoke 结果

状态：`FAILED_TO_PROMOTE_STOP_AT_5_UPDATES`
25-update / 长训：**锁定**

## 一句话结论

5% stateful recovery RSI 相比 0% 同步训练对照，确实恢复了起步语义并进一步减小平均航向偏差，但 source / f000 / f005 在当前严格 official matched-event 门下均为 `0/5`，且 f005 的停车漂移与持续后仰更差；这是局部信号，不是可晋级的 BASE 改善。

## 假设

在 Phase6 projected physical snapshot 与 Phase8 deterministic suffix 合同成立后，5% 精确 stateful recovery 初始化应扩大独立 stand/recovery backend 的恢复域，并且不破坏 source 或 0% 训练对照的官方全流程表现。

## 干预

- source：stand backend `model_150.pt`；
- f000：同 source、seed 47、5 PPO update，`recovery_fraction=0.00`；
- f005：除 `recovery_fraction=0.05` 外与 f000 相同；
- 两支均恢复 root/q/dq、episode clock、command、previous raw/issued action、phase 与 low-command moving latch；
- 训练只更新独立 93D→15D stand/recovery backend，Stage306 moving actor 完全冻结。

训练产物的 `env.yaml` 自查确认：除 recovery fraction 和日志路径外无其他配置差异。两份 ONNX 的 PyTorch 对齐误差分别为 `1.19e-7`、`1.79e-7`。

## 对照

source / f000 / f005 各跑 5 条 official AimDK v1.0 MuJoCo full episode：

```text
prepare 0.2 s → stand 2.0 s
→ matched start/move 5.2 s
→ curriculum stop 2.0 s
→ stationary/recovery handoff 8.0 s
```

固定：stiff `official_kp_ankle ×1.2`、upper fixed、无 supervisor、无 heading/cross-track action 修补、step clock。signed root pitch 只报告，不进 full gate。

## 结果

### 严格门

| backend | stand | startup | move | stop | full |
|---|---:|---:|---:|---:|---:|
| source | 5/5 | 5/5 | 0/5 | 0/5 | **0/5** |
| f000 | 5/5 | 0/5 | 0/5 | 0/5 | **0/5** |
| f005 | 5/5 | 5/5 | 0/5 | 0/5 | **0/5** |

15/15 条都是有效物理 episode，机器人没有因高度门提前倒地；失败来自语义/质量门，而不是 simulator、actor shape 或接口错误。

### 连续指标均值

| backend | heading max | 横向位移 | stop drift | stop settle | move signed pitch |
|---|---:|---:|---:|---:|---:|
| source | 0.628 rad | 0.551 m | 0.423 m | 3.748 s | -8.18° |
| f000 | 0.474 rad | 0.413 m | 0.441 m | 3.352 s | -7.32° |
| f005 | **0.428 rad** | **0.410 m** | 0.452 m | **3.336 s** | **-9.56°** |

f005 相对 source：

- 平均 heading max 改善约 31.8%；
- 横向位移改善约 25.5%；
- stop settle time 改善约 11.0%；
- stop drift 反而恶化约 6.8%；
- move 后仰从 -8.18° 恶化到 -9.56°。

f005 相对 f000：

- 恢复了 `startup 0/5 → 5/5`；
- heading max 再改善约 9.7%；
- 横向位移几乎持平；
- stop drift 恶化约 2.4%；
- 后仰明显恶化。

## 结论

假设只得到局部支持：stateful recovery 数据不是完全无效，它能抵消普通 5-update 对起步语义的破坏，并对航向/横漂产生正信号。但它没有让任何 episode 通过 move/stop/full gate，也没有改善停车位置锚定。

更关键的是，当前 full gate 的两个主失败量并不都属于 stand/recovery backend 的权限：

1. move 阶段始终由冻结 Stage306 actor 控制，recovery 数据只能通过 handoff 初态间接影响，不能根治 moving actor 的左右非等变和世界航向漂移；
2. stop backend 更快把速度降到零，却停在错误位置，说明现有 stand/recovery reward 更偏“速度归零/姿态稳定”，不足以学习世界水平位置锚定；
3. f005 加重 sagittal 后仰，证明把更多临界 stop 状态喂给 PPO 不会自动修复姿态平衡点。

因此，不能因为 f005 的 heading 连续值更好就宣称 recovery 路线成功，也不能用更长训练赌它自然跨门。

## 下一步

1. **停止 Phase9 于 5 updates**；不运行 25-update，不继续扫 recovery fraction；
2. source stand backend 仍是冻结部署基线，f000/f005 仅保留为机制证据，不替换 BASE；
3. move gate 回到已证实的 actor 左右非等变/方向合同主线；stand/recovery backend 不承担修复 moving actor 的任务；
4. 若未来重新训练 recovery backend，目标必须显式包含世界水平 stop-anchor，同时以 source policy anchor 保护起步，并把 signed pitch 作为预注册相对门；在此之前不值得继续 PPO；
5. 本轮没有多余中间 checkpoint：只生成并保留两支 final `model_155.pt`、ONNX 和 export manifest，无需删除用户旧资产。

## 产物

- 预注册：[x2_recovery_phase9_preregistration.md](x2_recovery_phase9_preregistration.md)
- 完整 15-run 面板：`reports/official_x2/phase9_stateful_recovery_paired_gate.json`
- 训练入口：`scripts/run_phase9_stateful_recovery_paired_u5.sh`
- official gate：`tools/official_x2/run_phase9_stateful_recovery_paired_gate.sh`
- f000 checkpoint SHA：`885b22227bf0550636ea5125a8f6139cbfb01b45de25739b49fcf1fa2de50085`
- f005 checkpoint SHA：`c2d7b87bb66443dfd2a90affe2bb4a384862388dc064c2adf6a5e75b428df144`
- f000 ONNX SHA：`20e1859a36760bf32f0c29c94c491c7ac6fab86f8fddd3b6fbe769ba3e636449`
- f005 ONNX SHA：`9bc672fc3c535dbe6cd2709cdec4531eb9b61990457172e9c813a8ac713fc0ca`

真机、WBT、Git、百度网盘均未触发。
