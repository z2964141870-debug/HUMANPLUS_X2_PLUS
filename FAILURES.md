# 失败与异常记录

## GPU NVML 版本不一致（2026-07-28）

- 内核模块：595.71.05
- 用户态 `libnvidia-ml.so` / `libcuda.so`：595.84
- 表现：`nvidia-smi` 报 `Driver/library version mismatch`
- PyTorch 曾报告 CUDA available/device count 1，但本轮 Isaac Sim 明确报告
  `no suitable CUDA GPU` 并进入 compatibility mode；不得据此声称 PhysX GPU 加速有效。
- 处理：未擅自重启或重装驱动；仿真仍能完成且零幅 A/B 逐数组完全一致，
  因此用于本轮确定性接口门禁，但不用于吞吐量或 GPU 性能结论。

## 单测解释器与初始 fixture（2026-07-28）

- 首次误用系统 Python，缺少 torch；改用 `x2-sonic-isaaclab`。
- 初始 YAML fixture 把 Hydra `_target_` 元数据误当 reward term；已过滤。
- 初始 synthetic reward 未乘真实控制步长，放大了 float32 重排误差；按 IsaacLab 0.02 s 积分尺度修正，运行时仍坚持 `1e-6` 硬门。

## Semantic dual critic 未形成物理 Pareto（2026-07-28）

- 正信号：E1 的两个 value head EV 为 `0.832/0.830`，优于 shuffled N1 的 `0.734/0.647`；训练 reward/episode length 也最高。
- 反例：E1 四域 stable 仅 `12/16`；wrist mean `71.07 mm`，相比 B0 `56.23 mm` 恶化 `26.38%`。
- 判定：这是“critic 估计改善但共享 actor 目标冲突仍存在”的有限正结果，不是长训解锁信号。
- 处理：停止扩大到 200×3，不用更长预算掩盖能力交换；转做 objective gradient conflict 诊断。

## 持续 Actor 梯度冲突假设未成立（2026-07-28）

- 边界矩阵：E1/N1 × INIT/I25 × 3 seeds 的 48 个 microbatch 全部正 cosine。
- 完整轨迹：E1 25轮共 300 microbatch，仅 2 个为负（0.67%），25/25 轮 aggregate gradient 全为正。
- 最小反例：即使 E1 wrist mean 已明显退化，两 head 在 I25 的三 seed aggregate cosine 仍为 `0.892–0.899`。
- 判定：PCGrad/GCR-PPO 所需的持续负冲突前提不存在；继续实现属于低价值方法堆叠。
- 更可能根因：目标分组含 whole-body compound term、upper 数据分布不足、训练 head reward 与 held-out wrist preservation 不等价。

## 腰腿动作投影未带来持续四域改善（2026-07-28）

- 5-step 正信号：LBP 在 delay 域将 root RMSE 从 `0.240` 降至 `0.176 m`，progress-ratio error 从 `2.870` 降至 `1.206`，contact 从 `2/4` 升至 `4/4`。
- 25-step 反例：LBP delay stable 从 `4/4` 退到 `2/4`，root RMSE 升至 `0.402 m`，出现两次末段跌倒；四域 strict 为 `0/16`。
- 训练表象：reward `0.551→4.546`、episode length `8.0→63.63`，但 reward-per-step 仅约 `0.063–0.071`，世界 anchor 误差不单调。
- 判定：动作投影只解决“新 LoRA 直接改哪些关节”，不能解决训练目标与世界步态门禁的错配，也不能阻止下肢失稳间接破坏手部任务空间位置。
- 处理：停止 200/1000-step；保留 LBP 作为安全 contract，转向 objective/evaluation 对齐。

## 原速/中幅上肢动作超出下层扰动裕量（2026-07-28）

- 0.25× 原速在 ideal/filter 中相对门通过，但 nominal+delay 的 lateral drift
  从 `0.612` 增至 `0.915 m`；机器人未摔，但控制契约不合格。
- 仅放慢时间轴到 0.5× 后，0.25× 候选的 lateral 降至 `0.505 m`、
  heading 降至 `0.491 rad`，预注册门通过。
- 将幅值提高到 0.50× 后，虽然仍走满 10 秒，但 heading 相对恶化
  `0.172 rad`、lateral 相对恶化 `0.467 m`，再次失败。
- 判定：失败来自有限扰动裕量和时序敏感闭环，不是“上肢完全不可分层”，也
  不能靠继续扫 PPO 轮数解释。
- 处理：冻结 0.25×/0.5×时间工作点；下一轮只做有界安全 Adapter 与覆盖扩展。

## 零手臂延迟反事实落入不同吸引域（2026-07-28）

- 去掉 arms 的 10.5 控制帧 delay 后，静态上身 control 在 4.78 秒跌倒；
  相同反事实下动态 0.25× 上肢候选反而走满 10 秒。
- 这使该组相对 lateral gate 失去可比性；不能得出“手臂零延迟更差”或
  “上肢动作治愈跌倒”的普遍结论。
- 处理：只将其记录为窄吸引域证据。正式归因依靠 ideal、filter、完整 delay
  与腰部反事实的共同模式。

## 有界上肢 Adapter 未形成跨速度安全包络（2026-07-28）

- 正常速度 `0.30 m/s`：swing/knocking/box_lift 通过，真实 wave 因 heading
  与 lateral 相对恶化 `0.201 rad/0.223 m` 失败。
- 低速 `0.20 m/s`：4 条均未通过，wave/swing 跌倒；control 自身虽生存，
  heading/lateral 已达 `0.715 rad/0.756 m`，说明基座余量很小。
- 所有目标均满足 `0.12 rad/0.20 rad/s` 边界，tracking p95 也小于
  `0.114 rad`；不能把失败归因于边界器失效或简单跟踪不上。
- seed42/7 逐项相同，说明当前无随机化 evaluator 的 seed 轴是假重复。
- 处理：不解锁 SONIC，不继续扫固定 filter；下一步需要下层的 anticipatory
  conditioning 或 disturbance training。

## IMU 航向回退产生反向安全效果（2026-07-28）

- guard 在低速 `1.66–2.56 s` 触发，并在 0.10–0.56 s 内将上肢目标平滑
  收回默认。
- wave/swing 仍跌倒；原本能走满 8 秒的 knocking/box_lift 也在约 5.66 秒
  跌倒，低速生存由 2/4 降为 0/4。
- 正常速度 wave 在 4.88 秒回退后仍未修复已积累的 heading/lateral。
- 判定：窄吸引域中“撤销上肢目标”不是状态回滚，而是另一段动力学输入；
  事后 supervisor 无法替代提前协调。
- 处理：保留代码和反例报告用于 future emergency guard 对照，但该机制不
  进入主线。

## Future-intent 出现均值改善但留出动作回归（2026-07-28）

- Gate5 FUTURE 的生存仍为 5/12，但平均 heading/lateral 优于 CURRENT 与
  FUTURE-NOPHASE，说明未来意图和 phase 不是完全无效。
- Gate25 FUTURE 的总生存步数为 3218，高于 BASE 3176；平均 heading/lateral
  降到 `0.560 rad/0.282 m`，正常速度的三个训练分布动作明显改善。
- 反例一：低速六动作总生存由 BASE 962 降为 945，主要失败带未修复。
- 反例二：留出 `wave_left@0.30` 的 heading 从 `0.510` 恶化到
  `1.068 rad`，tilt 从 `0.299` 升到 `0.509 rad`。
- 判定：Adapter 学到了动作子集相关协调，而非通用扰动映射；aggregate
  均值不足以覆盖逐案例安全回归。
- 处理：保留 checkpoint 和因果证据，不晋级、不继续同目标长训；下一设计
  必须支持风险约束或拒绝介入。

## 保守 Future Adapter 未形成多种子横向鲁棒性（2026-07-28）

- 固定面板正信号：NEUTRAL05 相比 BASE 总步数 `3176→3248`、低速步数
  `962→974`，heading/lateral 均改善；因此解锁随机初态复测。
- 多种子正信号：36 个成对扰动案例中生存 `15→17`、总步数
  `9674→9961`、世界 heading `0.5988→0.5697 rad`、tilt
  `0.5465→0.5111 rad`。
- 反例：聚合 lateral `0.4657→0.4872 m`，BASE 正常速度生存案例未全部
  保留，且三个 seed 都未同时满足 steps/heading/lateral 三项不劣。
- 删失审计：共同生存窗口 lateral max 略好，但 lateral 时间均值
  `0.178080→0.180017 m`；seed42 的共同生存 wave-real 横漂恶化
  `0.1513 m`，超过预注册门。
- 判定：Adapter 确实改变了闭环吸引域并救回部分案例，但仍以动作相关横漂和
  个别新增跌倒交换收益；不是可靠迁移模型。
- 处理：保留 Stage208 为正式基线、NEUTRAL05 为互补研究候选；删除失败的
  RISK05 checkpoint，不沿当前 objective 续训。
