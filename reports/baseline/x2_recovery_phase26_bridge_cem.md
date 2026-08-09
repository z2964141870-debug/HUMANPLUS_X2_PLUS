# BASE Phase26：Official-physics dynamic bridge teacher 最小验证

## 游戏任务

- [x] 审计并复用 Phase8 test-only official MJCF 的完整物理状态与 controller snapshot 分叉。
- [x] 冻结 Stage306 moving、Stage335 f005 recovery、source stationary、PD、上肢和 stop command。
- [x] 从同一个 stop-event snapshot 做 zero-residual control 与唯一一次低维 CEM smoke。
- [x] 只搜索 1.0 s 腰腿 bridge；交权后继续运行 f005 2.0 s，并以首 1.0 s 作严格门。
- [ ] 找到严格进入 success-safe 联合支持、交权后持续留域的 bridge。
- [ ] 解锁 PPO 或长训。

## 假设

Phase25 表明交权时完整 q/root 已不在 recovery 的闭环支持域，而实际 issued action 并未先稳定 OOD。若主要缺口是 brake 与 f005 之间缺少一个可达的动态过渡，那么在同一个物理初态上，短时低维腰腿 residual 应能把末端 q/root/IMU/action-history 推入 Phase19/20 的 `success_safe` 支持，并使 f005 接管后至少 1 s 不离域、不倒。

## 干预

冻结根部物理状态，不做 teleport，也不改模型、PD、上肢或 stop 命令。bridge 长度固定 1.0 s，用 5 个预注册的 normalized mode、每个 2 个 C2 knot：

1. sagittal crouch；
2. hip-roll anti-symmetric；
3. ankle-roll anti-symmetric；
4. waist pitch；
5. waist roll。

10 个变量均限于 `[-0.2, 0.2]`。CEM 固定 seed=2601、population=16、iterations=3、elites=4，总计 48 个候选；没有扫 horizon、维度、bounds、seed 或门限。

## 对照

对照和候选从完全相同的 Phase8 integration-state + controller snapshot 恢复：

- control：1.0 s 零 residual bridge，再以 0.5 s target blend 交给 f005；
- candidate：CEM 最优低维 bridge，再以相同 0.5 s blend 交给同一个 f005。

preflight 证明物理 snapshot bitwise exact、controller snapshot exact，两条独立 fork 的未来 10 tick 逐项 exact。control 与 candidate 都真实执行交权；“不交权”不能算成功。

## 结果

### 搜索结果

| 指标 | zero control | CEM best | 趋势 |
|---|---:|---:|---:|
| 预注册 cost | 4.912 | 3.254 | 改善 33.8% |
| bridge 末 root z / tilt | 0.632 m / 0.061 rad | 0.637 m / 0.055 rad | 小幅改善 |
| 交权后 1 s 最低 root z | 0.628 m | 0.633 m | 小幅改善 |
| 交权后 1 s 最大 tilt | 0.149 rad | 0.136 rad | 小幅改善 |
| 交权后 2 s 最大 tilt | 0.199 rad | 0.147 rad | 改善 |
| 严格 dynamic bridge | 否 | 否 | 未打通 |

CEM 三代 best cost 为 `4.128 → 3.254 → 3.397`；最终保留全局最优 3.254。两支在 2 s 报告窗口均未触发 root-z `<0.55 m` 或 tilt `>0.30 rad`，但 root safety 只是必要条件，不代表进入 recovery 支持域。

### 末端 success-safe LOO-p95 归一化距离

`<=1` 才在对应支持门内。

| 组 | zero control | CEM best | 是否通过 best |
|---|---:|---:|---:|
| base linear velocity | 0.271 | 0.288 | 是 |
| base angular velocity | 0.035 | 0.162 | 是 |
| projected gravity | 0.946 | 0.128 | 是 |
| joint position | 2.100 | 1.955 | **否** |
| joint velocity | 0.050 | 0.066 | 是 |
| previous action | 0.849 | 1.050 | **否** |
| root posture | 0.842 | 0.189 | 是 |

最优 bridge 显著把 projected gravity 与 root posture 拉回 success-safe 中心，也略改善 joint position；但 q 仍约为门限的 1.95 倍，previous-action 也刚刚越门。它没有完成联合状态桥接。

### 交权后首 1 s union-support 最大归一化距离

| 组 | zero control | CEM best | 是否通过 best |
|---|---:|---:|---:|
| base linear velocity | 0.759 | 0.465 | 是 |
| base angular velocity | 0.681 | 0.718 | 是 |
| projected gravity | 2.913 | 1.368 | **否** |
| joint position | 2.534 | 2.329 | **否** |
| joint velocity | 0.133 | 0.178 | 是 |
| previous action | 2.674 | 2.731 | **否** |
| root posture | 2.110 | 1.471 | **否** |

因此“交权后至少 1 s 留在联合支持域”明确失败。CEM 改善了 gravity/root 的离域幅度，但没有改变 q/action-history 的主要闭环缺口。

## 结论

这次最小验证得到的是**机制信号，而不是可用 bridge**：在诚实可复现的同初态 official-MJCF 分叉中，低维腰腿 bridge 能降低综合支持距离并改善姿态，但 48 个固定预算候选没有把末端 joint position 与 previous action 同时送入 success-safe，也没有让 f005 接管后的完整状态持续留域。

所以不能宣称动力学/reference 冲突已经解决，也不能解锁 PPO/25-update/长训。更不能把“2 s 没倒”当成成功；严格失败来自支持域条件，而不是二元倒地门。

这也不是对 dynamic bridge 路线的最终否定：单次 smoke 只验证了一个 stop-event、5 个手工低维 mode、1.0 s horizon 和 48 个候选。它只否定“当前这一版小型 bridge 已足够”的命题。

## 下一步

按预注册在此停止，不继续扫 CEM 参数。若未来继续，信息增益最高的唯一方向是先重构 bridge 表示，使其显式优化 joint-position 与 actual action-history 的可达性（并保留 root/gravity 约束），再做新的预注册实验；在此之前不应训练 recovery actor。

## 证据边界

- 使用未修改的官方 scene XML 与 MuJoCo 物理。
- 这是 Phase8 的**同进程 test-only official MJCF**，不是 AimDK ROS 完整闭环；公开 ROS simulator 没有无损 mid-event state injection。
- contact/COM 等量均为模型估计，不是真机传感器真值。
- 没有 PPO、optimizer、真机、WBT、Git 或百度网盘操作。

## 证据文件

- preflight：`reports/baseline/x2_recovery_phase26_bridge_preflight.json`
- CEM 结果：`reports/baseline/x2_recovery_phase26_bridge_cem.json`
- runner：`tools/official_x2/run_phase26_testonly_bridge_cem.py`
- tests：`tests/test_phase26_bridge_cem.py`
- frozen physical snapshot SHA256：`c59c7c95d88e0080ee28bfd35bbd06c3b55e26acb3e24cef7c1088280e441391`
