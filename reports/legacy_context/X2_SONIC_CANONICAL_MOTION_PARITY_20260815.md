# X2-Sonic canonical motion 分布与 observation parity 记录

日期：`2026-08-15`
模式：离线数据适配 + MuJoCo 闭环仿真；未连接机器人、Orin 或 BLE。

## 目的

验证“PHUMA/后续衣服动作先转换到官网 motion-bank JSON 契约，再做 observation parity 和闭环测试”这条路线，排除文件格式、关节顺序和 loader 差异对结论的影响。

## 工程位置

```text
项目根目录：/home/yu/projects/BFM-Zero
适配器：/home/yu/projects/BFM-Zero/tools/official_x2/canonicalize_x2_motion.py
单对 parity：/home/yu/projects/BFM-Zero/tools/official_x2/compare_x2_motion_contract.py
批量 parity：/home/yu/projects/BFM-Zero/tools/official_x2/batch_compare_x2_motion_contract.py
原始 PHUMA：/home/yu/projects/x2_teleop_final/x2_sonic/motion_lib_x2/
canonical motion bank：/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/canonical_motion_bank/phuma_diverse_5s_v3/
数据 manifest：/media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-15/phuma_diverse_5s_canonical_manifest_v3.json
```

原始 pkl 保持只读，canonical JSON、index 和报告全部写入 data mount；没有把大文件写入 Git 工作树。

## Canonical 契约

每个 pkl 被转换为官网 motion-bank 使用的 JSON 字段：

```text
name, display, fps, frames, root_pos, root_quat, joint_pos
```

转换步骤：

1. 使用现有 X2 PHUMA loader 校验 31 个 MuJoCo 关节及 `(T, 3)/(T, 4)` 根状态。
2. 使用官网等价的 heading normalization。
3. 重采样到 50 Hz；本批输入本来就是 50 Hz，因此没有改变采样点。
4. 保留运行时的一秒 stand blend 边界，不把 stand blend 写进 JSON，避免运行时重复混合。
5. 以“来源子集目录 + clip 名”命名，避免 strict/medium/broad/hybrid 同名文件互相覆盖。
6. 生成官网风格的 `index.json` 和逐文件 SHA256 manifest。

## 结果

### Observation parity

对 4 个 PHUMA 子集抽取的 12 个 clip，分别用原始 pkl 和 canonical JSON 经过同一个 X2-Sonic evaluator，并在相同的确定性 synthetic proprioception 状态序列上运行 50 tick：

```text
12/12 pass
observation 最大绝对误差：1.8533e-16
31-D action 最大绝对误差：0.0
```

批量结果：

```text
/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/contract_parity/phuma_diverse_5s_batch.json
```

### MuJoCo 闭环

当前 pkl 与 canonical JSON 都使用同一 ONNX、同一 MuJoCo scene、同一 CUDA provider、同一 5 秒测试协议：

```text
12 clips：7 通过，5 失败
失败原因：root_height_or_tilt
失败时间：约 1.08–1.88 s
```

canonical 结果：

```text
/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/official_sonic_x2_webscene_phuma_diverse_5s_canonical_v3.json
```

同批当前 pkl 对照：

```text
/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/official_sonic_x2_webscene_phuma_diverse_5s_pkl_current.json
```

按来源文件逐条匹配后，两者的通过/失败状态、失败 tick 和闭环指标一致到数值误差范围内。旧版报告的条目顺序不同，不能直接按数组位置比较；后续统一使用 manifest 的来源路径匹配。

## 研究解释

这次实验得到的是一个重要的排除性结论：

> PHUMA pkl → canonical JSON 的格式、关节顺序、heading、50 Hz 采样和 observation 构造均已对齐；canonical 化本身不会改善闭环动力学，也没有引入新的失稳。

因此，现阶段的 5 个失败样本不应再归因于“ONNX 文件不能直接用”或“JSON/PKL loader 不一致”。更可能的对象是：

- 动作本身超出 X2-Sonic 的训练分布；
- 根部速度、足部接触相位或躯干姿态与 X2 动力学不匹配；
- 目标动作的时间尺度/速度需要限速、平滑或相位重整；
- policy 输出在闭环中出现动作饱和，且小扰动会被动力学放大。

下一步应在 canonical motion bank 上加入**可追溯的动作分布适配层**，例如速度/加速度约束、根部相位处理和接触可行性筛选，然后用同一闭环协议比较“原始 canonical、适配后 canonical、衣服重建 canonical”三组，而不是立即重新训练 policy。

## 复现边界

- parity 是离线 deterministic synthetic-state 测试，不等价于真机安全。
- 闭环测试是官方 MuJoCo scene，不等价于 X2 真机结果。
- 本批只覆盖 12 个 PHUMA clip；扩大到整套 PHUMA 或衣服日志前，必须复用相同 manifest 规则。
- CUDA provider 依赖当前 conda 环境中 NVIDIA pip 库的进程级 `LD_LIBRARY_PATH`；没有修改系统 CUDA 或环境目录。
