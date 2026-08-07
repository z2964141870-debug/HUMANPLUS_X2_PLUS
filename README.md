# CWI Cross-Embodiment Simulation

这是 SONIC/G1 → AgiBot X2 跨具身控制迁移的独立仿真研究工程。旧 X2/SONIC
工程保持只读；本仓库保存新实现、实验预注册、结果卡、裁决报告与可复现脚本。

## 当前结论

- CWI 式多 critic 能改善价值估计，但没有形成 X2 物理性能 Pareto；
- 将腿腰控制权留给 X2 下层、上层只提供有界慢速动作意图具有局部可行性；
- Future-intent + gait phase 能跨随机初态改善聚合生存和航向，但横向鲁棒性
  尚未通过严格门；
- Stage208 仍是正式基线，NEUTRAL05 只保留为研究候选；当前 Future-Adapter
  objective 已停止。

优先阅读：

- [`TASK_CARD.md`](TASK_CARD.md)：研究目标和边界；
- [`STATUS.md`](STATUS.md)：逐阶段状态；
- [`ROUND7_RESULT_CARD.md`](ROUND7_RESULT_CARD.md)：当前最新裁决；
- [`DECISIONS.md`](DECISIONS.md)：不可事后修改的关键决策；
- [`FAILURES.md`](FAILURES.md)：失败假设与反例；
- [`history/legacy_x2_migration/INDEX.md`](history/legacy_x2_migration/INDEX.md)：Stage 0–163 旧迁移历史、关键报告与完整归档恢复入口；
- [`ARTIFACT_STORAGE.md`](ARTIFACT_STORAGE.md)：大文件存储与恢复规则。

## 仓库边界

普通 Git 只保存源码、配置、测试、报告和小型实验证据。以下内容故意不进入
Git：

- `logs/`；
- `checkpoints/`；
- 视频、数据集、原始机器人/仿真 trace；
- `*.pt`、`*.ckpt`、`*.onnx` 等模型二进制。

这样可以避免首次提交把约 2.2 GB 的本地实验目录复制进 `.git`，造成磁盘和
远端仓库同时膨胀。

## 基础验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:tools:/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox \
conda run --no-capture-output -n x2-sonic-isaaclab python -m pytest -q
```

完整 IsaacLab 运行仍依赖本机旧工程、X2 资产和 `x2-sonic-isaaclab` 环境；
仓库本身不复制这些第三方/旧工程资产。
