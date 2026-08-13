# X2 Cycle20 恢复卡（2026-08-11）

## 归档状态

- 本轮范围：Phase19–23 动态接触可行化、BASE Phase40 成功盆地、Phase41 原生边界及对应测试。
- 本地包：`/home/humanplus/projects/ZHY/backups/x2_cycle20_contact_skeleton_20260811.tar.gz`
- 大小：`46,608 B`
- SHA-256：`6133637f79251ccf1d7e9a03a464bfcd9978bd1415f678dc987d1b4c10a30fab`
- 百度路径（相对 `/apps/bdpan/`）：
  `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-11/cycle20/x2_cycle20_contact_skeleton_20260811.tar.gz`
- 上传状态：官方 `bdpan` 命令成功；远端下载 SHA 抽检待周期人工审计。
- 上传 manifest：`docs/backup/manifests/6b22c48f81b81b164c48a9d554899cf7b1744a6a3b78e52c00359c3752e1ca40.json`

## Git 边界

- 打包时仓库 HEAD：`ac4d34f97a461b3e5c61bbec754ddef5f89449da`。
- Phase19–23、BASE Phase40/41 当前由本归档保存，但尚未进入 Git commit。
- 原因：此前 Git commit/push 的授权确认未获得明确同意；不得把百度上传成功冒充 Git 已保存。

## 恢复

1. 克隆 Git 仓库并检出至少 `ac4d34f97a461b3e5c61bbec754ddef5f89449da`。
2. 下载本包到仓库父目录。
3. 校验：

   ```bash
   sha256sum x2_cycle20_contact_skeleton_20260811.tar.gz
   ```

4. 在 `CWI_CrossEmbodiment_Sim` 根目录解包：

   ```bash
   tar -xzf x2_cycle20_contact_skeleton_20260811.tar.gz
   ```

5. 运行纯回归：

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
     python -m pytest -q -p no:cacheprovider \
     tests/test_phase19_hard_contact_preflight.py \
     tests/test_phase20_temporal_backend.py \
     tests/test_phase20_temporal_hard_contact_contract.py \
     tests/test_phase20_temporal_hard_contact_result.py \
     tests/test_phase21_contact_keyframe_skeleton_contract.py \
     tests/test_phase21_contact_keyframe_skeleton_result.py \
     tests/test_phase22_sequential_contact_skeleton.py \
     tests/test_phase23_time_parameterized_skeleton.py \
     tests/test_phase40_stage250_success_basin.py \
     tests/test_phase41_native_boundary_seed.py
   ```

预期：`15 passed`。

## 当前技术裁决

- Phase22：七段离散接触骨架硬可行。
- Phase23：普通 joint/root minimum-jerk 插值产生 `-3.483 mm` 中间穿透，时间化轨迹拒绝。
- 下一技术层：转换段逐帧保持支撑足 contact manifold；禁止仅调慢时间或直接进入 physics/RL。
