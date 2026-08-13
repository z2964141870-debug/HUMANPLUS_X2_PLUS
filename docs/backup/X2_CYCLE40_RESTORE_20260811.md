# X2 Cycle40 恢复卡（2026-08-11）

## 归档

- 范围：Phase32–35 原生稳定盆地 DSMS；Phase36–39 privileged physical-generator seed/hook/reward/exporter/reset batch。
- 本地包：`/home/humanplus/projects/ZHY/backups/x2_cycle40_native_dsms_generator_20260811.tar.gz`
- 文件数：`43`
- 大小：`741,458 B`
- SHA-256：`680500a8b6f07f7a2b02c630747ece0b71099bf705d228c822f113da519d205e`
- 百度相对 `/apps/bdpan/`：`HUMAN+/HUMANPLUS_X2_PLUS/2026-08-11/cycle40/x2_cycle40_native_dsms_generator_20260811.tar.gz`

## Git 边界

- 打包时 HEAD 仍为 `ac4d34f97a461b3e5c61bbec754ddef5f89449da`。
- Cycle20/30/40 尚未 Git commit/push；此前授权未明确，不得写成 Git 已同步。
- 百度包独立可恢复，不依赖未提交工作树。

## 恢复与验证

```bash
sha256sum x2_cycle40_native_dsms_generator_20260811.tar.gz
tar -xzf x2_cycle40_native_dsms_generator_20260811.tar.gz

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_phase32_native_dsms_preflight.py \
  tests/test_phase33_native_control_spline.py \
  tests/test_phase33b_native_pd_spline.py \
  tests/test_phase34_native_dsms_nlp.py \
  tests/test_phase35_native_dsms_short_solve.py \
  tests/test_phase36_native_generator_seed.py \
  tests/test_phase37_native_generator_hook.py \
  tests/test_phase38_privileged_generator_readiness.py \
  tests/test_phase39_privileged_generator_reset_batch.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
conda run -n x2-sonic-isaaclab python -m pytest -q -p no:cacheprovider \
  tests/test_x2_privileged_generator_io.py
```

## 当前裁决

- 已证明 Phase34 stable walk 是 DSMS 可精确恢复的成功动力学盆地；340ms qpos/qvel-only replay 误差约 `1e-13`。
- 18-knot torque/PD target 都保持安全，但不能逐节点复刻高频 qvel；动态一致 shooting node 的 defect 达到 `7.8e-15`。
- 唯一 Phase35 50-iteration DSMS 未收敛，0ms swing/0mm clearance，按门拒绝；不扩预算。
- 学习式路线已导出 200×50Hz native state-only seed，WBT 10×58 future 与 31/29 reset adapter通过。
- privileged-generator 底层合同 6/6；realized-physics reward 与 rollout exporter已实现。live Isaac reset wiring和live-zero manifest仍未完成，训练保持锁定。

## 外部依赖

本包不重复包含 Phase34 约 595MB closed mmap；恢复完整运行仍需 Cycle30 恢复卡所列 official X2 assets，以及：

`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase34_full_stage250_trace/phase34_stage250_closed_full_once_d230.mmap`

其 SHA-256 必须为 `8bb90c96a17308ce6f3e5be1f19f72a3e3998800f29d6f020692eac318de24ef`。
