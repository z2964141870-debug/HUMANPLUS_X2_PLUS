# Scalar/Vector Reward 等价性测试方案（Phase 1 设计稿）

日期：2026-07-26（Phase 0 产出，Phase 1 实施）
前置事实：见 reports/phase0_reward_audit.md §3、manifests/baseline_manifest.yaml code_paths。

## 1. 实现方案（overlay，不改旧工程源文件）

### 1.1 挂钩点

IsaacLab `RewardManager.compute()` 已在 `self._step_reward` 保留逐 term 值
`(num_envs, num_terms)`（reward_manager.py:156，单位：term_raw × weight × dt 之前
的分量——实施时须核对 _step_reward 存的是否已含 weight×dt，若是原始值则分组端
乘以同一 weight×dt）。

vector reward 生成：在 gear_sonic 的 env wrapper 层（manager_env_wrapper.py step()
返回处，line 975 附近，通过 overlay 子类实现，不改原文件）读取
`env.reward_manager._step_reward` 与预注册 grouping mask（term→group 的 0/1 矩阵 M，
形状 (num_terms, K)），输出 `r_vec = _step_reward @ M`，形状 (num_envs, K)。

trainer 侧：ppo_trainer.py line 10547-10554 已支持 (num_envs, K) 直存；
config `rewards.num_critics: K`。

### 1.2 Overlay 方式

新建 `src/dcpeft/` 包：子类化 wrapper（或 monkeypatch 注册），配置经新的 exp_config
YAML 引用；旧工程文件零修改。若最终发现必须改旧文件，先在 patches/ 存
diff + 原文件 SHA256（任务卡 §3）。

## 2. 等价性测试（Contract 硬门）

### 2.1 静态划分检查（无需仿真）

- 每个 active term 恰属一组：`M.sum(dim=1) == 1` 对所有 active term；
- inactive（weight=0）term 显式列入 excluded 清单；
- grouping 与理由写入 manifests/reward_grouping_<G>.yaml，含 config SHA256 绑定。

### 2.2 动态逐步等价（fixed rollout）

流程：
1. 以 B0 checkpoint、固定 seed、deterministic rollout 运行 N=2 episodes × 4 motions
   ×（≥260 步）；
2. 每步同时记录：env 原生 scalar reward `r_s`、逐 term `_step_reward`、
   分组和 `r_vec.sum(-1)`；
3. 断言逐环境逐步 `max |r_s − r_vec.sum(-1)| ≤ 1e-6`；
4. 同时保存 rollout（obs/action/reward 序列 + SHA256）到 results/fixed_rollout/，
   供 N1 预注册与 Phase 3 回放复用。

### 2.3 广播防呆检查

- 断言 `r_vec` 各列不全等（K 列相关系数 < 0.999，防止同一 scalar 广播 K 次）；
- 断言 storage 中 rewards/values/returns/advantages 的 trailing dim == K 且
  各列统计不同；
- num_critics=1 回归：同一 rollout 下新路径与旧路径 total reward 逐位一致。

### 2.4 统计输出（供分组定稿）

fixed rollout 上输出各 term/各候选组的：mean、std、数量级、组间 Pearson 相关、
与 termination 的时序关系。写入 results/phase1_reward_stats.json。
G-A vs G-B 定稿规则（预注册）：
- 若 4 个全身模仿项与 loco 组相关系数中位数 ≥ 与 upper 组相关系数中位数 + 0.2，
  判入 loco（G-A）；反之判入 upper（G-B）；
- 相关性无明显差别时默认 G-B（贴近 CWI task/style 语义，且为 triple 剥离做准备）；
- 定稿后写入 DECISIONS.md，训练开始后禁改。

## 3. 通过条件（Phase 1 门禁，任务卡 §9）

- [ ] 2.1 静态检查全过；
- [ ] 2.2 max abs err ≤ 1e-6 逐环境逐步成立；
- [ ] 2.3 防呆检查全过；
- [ ] fixed rollout 完整可回放（两次回放 action 序列逐位一致）；
- [ ] N1 分组已预注册（固定 seed，写入 manifest）。
