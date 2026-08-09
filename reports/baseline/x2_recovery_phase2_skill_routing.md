# X2 BASE Recovery Phase 2：参数保护与技能交权裁决

日期：2026-08-09

状态：**全局 actor anchor 淘汰；stand/recovery 职责分离有局部价值但未提高可靠停车率；手写紧急阈值淘汰。**

## 实验 A：固定 actor anchor

### 假设

Stage337-f010 在5个 update 达到 `4/5`、继续到10个 update 退化为 `1/5`，可能来自 actor 累积漂移；固定源 actor 的二次参数锚应保留 recovery 收益并保护原闭环。

### 干预 / 对照

- 对照：Stage337-f010，源 `model_150`、seed47、10% recovery reset、5 update、无 anchor。
- 干预：其余完全相同，只设 `actor_anchor_coeff=1.0`；不是系数扫描。
- 官方门禁：AimDK v1 MuJoCo、stiff1.2、fixed upper、同一 Stage306 moving actor，5次完整事件。

### 结果

- actor 参数 RMS 漂移：`0.002374 → 0.001425`，减少约40%；anchor 确实生效。
- 官方 full gate：Stage337 `4/5`，Stage341-anchor `0/5`。
- Stage341 有4次 stop fail，其中3次实际倒地；move 也只有1/5通过。

### 结论

全局参数锚保护了“旧参数”，却同时抑制了 recovery 所需的新控制行为；参数距离不能代理闭环能力。停止 anchor 系数扫描，Stage341 不晋升。

## 实验 B：stand / recovery 技能职责分离

### 假设

同一个 stationary actor 同时负责开局正常站立与停车后的恢复，会把两个吸引域混在一起；应由冻结 source stand 负责初始站立，Stage337-f010 只负责 brake 后 recovery。

### 干预 / 对照

- 新增独立 `recovery_model` 会话与 last-action history。
- `stationary_model=source stand` 只用于开局 stand。
- `recovery_model=Stage337-f010` 只在 brake-to-policy latch 后使用。
- moving actor、PD、Future-intent、brake、blend 和全部门槛不变。

### 结果

- Stage343 full gate：`4/5`。
- move gate：`5/5`，五次 heading max 均小于0.3 rad。
- stop gate：`4/5`；唯一失败在 recovery latch 之前已经出现倾角发散。
- runtime inference 计数证明每次均同时调用 source stationary 与 recovery 两个独立 slot，不是报告标签伪分离。

把 Stage343 与 Stage344 中**未触发紧急规则、因此控制语义相同**的3次额外运行合并后，双技能原契约为 `5/8`，与 source 的 `3/5` 成功率接近。它稳定修复的是初始站立造成的行走航向分叉，尚未证明提高停车成功率。

### 结论

职责分离值得保留为架构能力，但不能用单轮 `4/5` 宣称晋升；`BASE_TRANSITION` 仍不替换。

## 实验 C：倾角紧急 latch

### 假设

Stage343 唯一失败在 stop 约1秒出现 `tilt≈0.194 rad`，而通过样本同刻约 `0.065–0.110 rad`；若在 `t≥0.8 s、speed≤0.10 m/s、tilt≥0.15 rad` 时绕过 double-support 等待并单向交给 recovery，可能救回该分叉。

### 结果

- Stage344 full gate：`1/5`。
- 2次实际触发 emergency latch，2次均未救回。
- 其余3次未触发，其中只有1次通过；说明普通路径本身仍有显著随机分叉。

### 结论

“检测到倾倒再提前切 recovery”太晚，而且 recovery actor并不具备从该动量状态恢复的证据。该功能默认关闭，仅保留为负实验复现；不再扫阈值。

## 当前诚实裁决

- 最佳研究 checkpoint 仍是 Stage337-f010/model_155，但没有可靠晋升。
- 双技能接口保留，因为它把 move gate 做到5/5并显式分离职责；尚不能视为完整后端。
- 当前 stiff-fixed 的根因已进一步定位到：**brake 过程中、recovery latch 之前的接触/动量分叉**，而非单纯 stand actor 参数漂移。
- 下一次 BASE 训练若继续，必须把 `move→decelerate→contact event→stop` 作为同一 episode 的训练事件；只从静态停车末态训练 stand/recovery，或继续加部署阈值，信息增益已低。
- 长训、完整81格矩阵和真机仍锁定。

## 证据

- anchor 5-run：[stage342_recovery_f010_anchor1_u5_gate.json](../official_x2/stage342_recovery_f010_anchor1_u5_gate.json)
- dual-skill 5-run：[stage343_dual_stand_recovery_f010_u5_gate.json](../official_x2/stage343_dual_stand_recovery_f010_u5_gate.json)
- emergency latch 5-run：[stage344_dual_recovery_emergency_latch_gate.json](../official_x2/stage344_dual_recovery_emergency_latch_gate.json)
- Stage341 checkpoint SHA-256：`a0e59748a9b7e73af70fa4016482d5b7cd338ce5c428e291c92d28ffa9d5b01b`
- Stage341 ONNX SHA-256：`e489dfff9838a0f2225bb70200bd1c43da514b069c63ae793c9a6671f4d518f5`
