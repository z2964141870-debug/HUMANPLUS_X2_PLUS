# X2 Cycle30 恢复卡（2026-08-11）

## 归档

- 范围：Phase24–31 连续接触/载荷/force preflight，BASE Phase42/43 原生边界选择及测试。
- 本地包：`/home/humanplus/projects/ZHY/backups/x2_cycle30_load_force_boundary_20260811.tar.gz`
- 大小：`89,040 B`
- SHA-256：`9efdcc638c14900eef2bf4edb35af43e00e56497af361cc3e6da3000e5f3b7c3`
- 百度相对 `/apps/bdpan/`：
  `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-11/cycle30/x2_cycle30_load_force_boundary_20260811.tar.gz`
- 上传 manifest：`docs/backup/manifests/1f0b75a5c9549a8f0df8ed7ab355c838894b5556fc7e610fda12888f6618a3a6.json`
- 状态：官方 `bdpan` upload 成功；下载 SHA 抽检待周期人工审计。

## Git 边界

- 打包时 HEAD 仍为 `ac4d34f97a461b3e5c61bbec754ddef5f89449da`。
- Cycle20/30 代码和报告已由百度独立包保存，但尚未 Git commit/push。
- 原因：此前 Git 提交授权没有得到明确确认；不得把网盘归档写成 Git 已同步。

## 恢复与验证

```bash
sha256sum x2_cycle30_load_force_boundary_20260811.tar.gz
tar -xzf x2_cycle30_load_force_boundary_20260811.tar.gz
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_phase24_contact_manifold_path.py \
  tests/test_phase25_load_transfer_skeleton.py \
  tests/test_phase26_centered_load_transfer.py \
  tests/test_phase27_left_half_continuous.py \
  tests/test_phase28_left_half_centroidal_force.py \
  tests/test_phase29_force_time_dilation.py \
  tests/test_phase30_quiescent_load_transfer.py \
  tests/test_phase31_balanced_boundary_geometry.py \
  tests/test_phase42_native_quiescent_boundary.py \
  tests/test_phase43_native_balanced_boundary.py
```

## 当前裁决

- Phase27：Stage250→左加载→右摆→落地的连续几何路径通过。
- Phase28：centroidal force 仅 `22/98` 帧可行。
- Phase29：2×放慢提升到 `33/98`，仍失败，停止时间扫描。
- Phase42/43：低速边界有效，但冻结 `COM=foot center` 几何目标导致 ankle limit；停止边界扫描。
- 下一路线：force/COP/centroidal-aware 生成或 Stage250 原生边界上的小规模 DSMS；不再事后修 COM 或扫时间。
