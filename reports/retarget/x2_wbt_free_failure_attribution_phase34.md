# X2 WBT Phase34：Phase33 free-root失败只读归因

## 裁决

- **最强证据指向reference接触可行性缺口，裸PD缺闭环是并行成立但尚不能定主次；没有证据证明数值reset错位是主因。**
- 本阶段未运行physics/optimizer；只读取Phase33 aggregate，并对冻结source做静态FK/连续性审计。
- Phase33没有保存逐帧trace，故不能伪造0–0.575s曲线或prescribed同窗指标。

## reset mismatch

- runner代码在两栏都显式把frame0 q/dq/root pose/root velocity写入sim；frame0 root速度为 0.1017m/s、角速度 0.4626rad/s，dq max 1.6959rad/s。它不是静止reset，但不是数值漏写。
- frame0静态active-sole contact为 L=True / R=False；reference与sim初始化使用同一scene/q/root，因此未发现几何frame0错位。
- solver/contact warmstart与原controller hidden state确实缺失，所以reset动态上下文仍是可疑项；现有aggregate不能量化它的影响。

## reference contact / pose feasibility

- 全5.8s reference：flight=0.973、SS=0.027、DS=0.000；右脚active sole从未接触，左脚仅 0.027。这是动态lunge的严重接触可行性警报。
- 0–0.56s source连续：q-step p95/max=0.0225/0.0495rad，root-step max=0.0046m，未见开局离散跳变。
- 同一free存活窗内，reference contact L/R=0.071/0.000，realized L/R=1.000/0.929，agreement=0.071。仿真迅速落到近双支撑，而reference几乎全flight。
- contact均为模型碰撞标签，不是实机GRF/COP/wrench；但已足以否定‘reference提供了清晰可执行支撑时序’。

## 裸PD闭环能力

- prescribed完整跟踪通过：q RMSE=0.0477rad、body pos/ori p95=0.0813m/0.3080rad。
- free于 0.575s触发fall；最后已保存控制帧的root pos RMSE/final=0.1015/0.1592m、orientation-error p95/max=0.5039/0.5537rad、sampled tilt max=0.8803rad、z min=0.5166m，torque sat=0.0000。
- 因termination发生在控制采样之间且terminal state未保存，现有证据不能区分最终是root-z还是tilt先越门；也没有prescribed 0–0.575s同窗trace。
- prescribed靠外力逐物理步固定root，free只剩无状态位置PD；因此它证明PD可跟关节，却不证明能生成载荷转移/平衡闭环。

## 三类归因

| 假设 | 当前证据 | 裁决 |
|---|---|---|
| 数值reset mismatch | q/dq/root pose/velocity显式一致；frame0几何contact同合同；但无solver/contact warmstart和controller hidden state | **未证实为主因，保留次要不确定性** |
| reference接触可行性 | 97.25% flight、右脚0接触；free迅速近双支撑，agreement 0.071 | **强支持存在系统性缺口** |
| 裸PD缺闭环 | prescribed全程通过而free 0.575s失败；无策略/历史/载荷反馈 | **强支持，但与reference缺口尚纠缠** |

## 唯一下一实验

只新增一次 **free-root exact-joint kinematic oracle**：同一Phase30 candidate、同一frame0 q/dq/root、同一official scene/50Hz path，逐物理步精确施加reference q/dq但保持root完全自由；与已存在的Phase33 free裸PD结果比较，不改reference、PD、root、不加warmup。若oracle仍在相近时刻倒下且contact仍与reference冲突，优先证伪‘只是PD跟踪误差’并锁定reference/contact/reset可行性；若oracle显著存活且contact对齐，则说明裸PD/闭环不足是主导。

该oracle只用于归因，不是可部署控制器、不是Gold晋升、不是训练。不得同时调PD、加warmup或修reference。
