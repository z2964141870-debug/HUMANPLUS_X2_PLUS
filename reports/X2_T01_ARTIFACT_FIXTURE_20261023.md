# T01/P0 工件固定 + 观测契约登记（2026-10-23）

本轮唯一目标：只读盘点 v0.2 复刻所需的工件与契约，登记防丢失。未触碰控制链。

## 交付
1. **模型台账**：data/X2_V02_ARTIFACT_MANIFEST_20261023.json（7 个 onnx 全量 SHA-256）
   - v0.2 路线主 actor = x2_sonic_s1ft16000_g1.onnx (7dfdec7c…)，即 run14/15 500Hz 真机通信验证所用模型
2. **部署源码快照**：deploy_ws/agi_x2_deploy_onnx_ref/（自 SoC1 ws/src 拉回，含 --writer-hz 补丁，hash 5 处）
   - 此前该源码仅存 SoC1；备份 sha256 对应二进制 466a9864…
3. **观测契约**：31 关节映射 leg[0,12)/waist[12,15)/arm[15,29)/head[29,31)；
   obs 布局真源 = deploy_ws/.../config/obs_config_x2_ultra.yaml（Phase 0 切片校验过）；
   50Hz 控制拍 / 500Hz writer / ZMQ 5558 输入、5557 调试
4. **安全参数固化**：action-clip、各组 target-dev、LPF、kp/kd scale 全表（见 manifest contract.safety_driver_args）

## 附带登记（治理表同步）
- 通信基座修复（500Hz 契约恢复）：run14/15 两轮 0 strikes，已完成待审核
- 报告 SONIC_V02_KEYBOARD_WALK_REVIEW_20261019 §9"本机只读"交付项：本轮覆盖 1/4（工件固定+契约），仿真隔离（T02）与受支撑真机（T03）未执行

## 未做/未知
- g1_token/dual/v16ft8 家族未上机，无真机证据
- KPlanner 键盘意图脚本原件（keyboard_walk_publisher.py 等）仍未收到
