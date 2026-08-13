# Phase26 — X2 中心卸载载荷转移骨架

## 裁决

`CENTERED LOAD TRANSFER MECHANISM CONFIRMED / FULL SKELETON NOT CERTIFIED`

Phase26 相对 Phase25 的唯一变量，是在左加载与右加载之间插入冻结的 Phase41 原生 COM 投影作为双支撑卸载中心；数值边界、接触、净空、solver 和真值声明均未改。

## 结果

- `DS_CENTER → DS_LOAD_LEFT → L_SUPPORT_R_SWING → DS_R_TOUCHDOWN_LEFT_LOADED`：全部硬可行。
- 新增 `DS_UNLOAD_LEFT_TO_CENTER`：硬可行，等式残差 `1.62e-10`。
- `DS_LOAD_RIGHT`：硬可行，等式残差 `1.41e-10`。
- `R_SUPPORT_L_SWING`：严格失败；等式残差 `4.23e-7` 已通过 `1e-6` 门，但最坏不等式 `-1.292e-7` 未通过固定 `-1e-8` 门。

运行完成 7/9 个关键帧，墙时 `2.11 s`，0 `mj_step`、0 GPU、0 训练。

## 解释

中心卸载态消除了 Phase25 从左加载直接跨到右加载的 `73 μm` 残差，说明载荷转移序列修正有效。当前失败缩小到右摆脚帧的 `0.119 μm` 容差超限，接近数值边界，但没有正式可行证书：不能把 near-miss 写成通过，也不能临时放宽容差、增加迭代或重启 solver。

这仍只是 COM 几何代理，不是 COP/GRF/动力学 teacher。后续若继续，应新立一个严格的对称初始化/数值证书假设，或仅在已证明的左半周期上进入 centroidal force preflight；不得直接运行 physics/RL。

## 产物

- `phase26_centered_load_transfer_contract.json`
- `run_phase26_centered_load_transfer_skeleton.py`
- `phase26_result.json`
- `tests/test_phase26_centered_load_transfer.py`
