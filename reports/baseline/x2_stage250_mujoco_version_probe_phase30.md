# BASE Phase30：MuJoCo 3.3.7 / 3.4.0 单变量 probe

日期：2026-08-09
裁决：`VERSION_IS_NOT_MAIN_CAUSE`
边界：唯一 0.2 秒 direct probe；无训练、无 closed ROS 新回放、无控制器改动、无 WBT/Git/百度/真机操作。

## 游戏任务卡

- [x] 冻结 Phase28 scene、stand 初态、前 10 个 action、31 关节 PD target、PD law、20 ms control cadence。
- [x] 绕开 ONNX Runtime：直接重放 Phase28 已保存 action/target，防止引入第二变量。
- [x] `LD_PRELOAD` closed wrapper 的**同一个** `libmujoco.so.3.3.7`，并校验映射路径和 SHA256。
- [x] 仅运行 10 control ticks / 200 physics substeps / 0.2 秒。
- [x] 按预注册三个主指标裁决；未扫 prepare、异步时序或其他参数。
- [ ] MuJoCo 版本主因：被严格排除。

## 假设

若 Phase28 的首 20 ms 速度爆炸主要来自 Python direct 使用 MuJoCo 3.4.0，而 closed wrapper 使用 3.3.7，那么把唯一变量换成同一份 vendor 3.3.7 后，应显著接近 Stage250 historical。

预注册主指标：

1. `t=0.02 s` joint dq historical abs-max error；
2. `t=0.02 s` root angular velocity historical abs-max error；
3. `t=0.06 s` root tilt historical absolute error。

预注册晋级规则：三个误差必须**全部至少缩小 50%**，才将版本视为主因；否则排除并停止。

## 干预

唯一干预：

```text
Phase28 control: Python MuJoCo 3.4.0
Phase30 candidate: exact vendor libmujoco.so.3.3.7
```

冻结合同：

- scene SHA：`7fceb3e1...bb1b63`；
- historical trace SHA：`310924d7...30c64`；
- Phase28 control SHA：`9abc1574...b0f40a`；
- Phase28 3.4 substep SHA：`2581e822...ecaab`；
- vendor 3.3.7 library SHA：`9bfb4d37...62ff97f`；
- runner SHA：`dca60394...1c946`。

Candidate 进程只映射：

`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/lib/libmujoco.so.3.3.7`

## 对照

不重新生成 3.4 control，复用 Phase28 已冻结的前 200 个 substep。Candidate 使用同一 visible qpos/qvel、同一 10 行 final action、同一 10 行 PD target，并按相同 PD law 在每 20 ms 起点计算 torque、保持 20 个 1 ms substep。

这意味着 ONNX、reference、action history、策略随机性均未参与本次比较。

## 结果

| 主指标 | Phase28 3.4 | Vendor 3.3.7 | 相对缩小 |
|---|---:|---:|---:|
| 0.02 s joint dq error | 4.437503 rad/s | 4.437503 rad/s | 0% |
| 0.02 s root angular velocity error | 2.518094 rad/s | 2.518094 rad/s | 0% |
| 0.06 s tilt error | 0.053095 rad | 0.053095 rad | 0% |

更强的结果是：整个 0.2 秒内，3.3.7 与既有 3.4 trajectory 的：

- joint q max difference：`0.0`；
- joint dq max difference：`0.0`；
- root position/velocity difference：`0.0`；
- quaternion 数组逐元素相同；由浮点 dot/acos 计算的 geodesic 数值上限仅 `2.98e-8 rad`。

所以不是“改善不明显”，而是该冻结合同下两个版本产生了相同的物理轨迹。

## 结论

`VERSION_IS_NOT_MAIN_CAUSE`

MuJoCo 3.3.7/3.4.0 版本差异不是 Stage250 closed success 与 Phase28 direct failure 的主因，已被单变量 probe 排除。Phase29 剩余的高信息候选收窄为：

1. closed wrapper 的 reset → 0.2 秒 prepare 所形成、Phase28 没有恢复的 contact/constraint/warmstart/control history；
2. closed ROS 的异步 command 到达与 1 kHz joint / 500 Hz IMU measurement scheduling。

二者当前仍被闭源边界绑定在一起。不能因此解锁训练，也不能改扫 prepare 时长或 delay。下一份真正有区分力的证据仍应是 closed wrapper 在 stand `t=0` 的 full integration state 与命令落步记录。

## 下一步

本阶段按任务卡停止。若后续获得新授权，下一实验必须一次只处理“prepare integration state”或“command scheduling”之一；在此之前保持：

- BASE 训练锁定；
- WBT native dynamic seed 锁定；
- Stage250 historical 能力证据保留；
- Phase28 direct failure 仅作为 domain-mismatch 证据。

## 产物

- 预注册：`reports/official_x2/phase30_mujoco_version_probe_prereg.json`
- 结果 JSON：`reports/official_x2/phase30_mujoco_version_probe.json`
- 运行工具：`tools/official_x2/run_phase30_mujoco_version_probe.py`
- 纯测试：`tests/test_phase30_mujoco_version_probe.py`
- 0.2 秒 candidate cache：外部 cache `phase30_mujoco337_probe/stage250_prefix_0p2s.jsonl.gz`
