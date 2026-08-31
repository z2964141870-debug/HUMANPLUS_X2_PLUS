# X2 SmartWear 新遥操流程（2026-08-14）

## 当前硬件与软件事实

- 新上衣 MAC：`D5:F4:A2:41:93:4B`
- 新裤子 MAC：`F3:FB:AD:FC:7D:82`
- 单设备 V2 数据可以正确解码，当前新固件的正常量级约为 `66–67 Hz`，旧流程中的“50 Hz”不再是准确门槛。
- 双设备同时连接时，本次实测上衣约 `66–67 Hz`，裤子约 `38–41 Hz`；裤子链路尚未达标。
- 两件衣服均未通过 UTC 同步门禁，程序停在“等待衣服 UTC 时间同步”。
- PC2 当前：
  - `sudo -n hcitool con` 会要求密码，自动低延迟参数无法执行；
  - `chronyc` 未安装；
  - `ble-time-bridge` inactive；
  - 时间桥默认 Python 路径不存在。
- 当前没有衣服遥操、BLE 测试或时间桥进程运行。

## 0. 一次性部署门禁

完成以下条件前，不进入真机遥操：

1. 为 `hcitool lecup` 配置最小范围的免密 sudo，不能给整个 sudo 放权。
2. 安装并确认 Chrony 已同步到有效 UTC 源。
3. 用 PC2 实际存在的 `/usr/bin/python3` 安装 `ble-time-bridge`。
4. 时间桥白名单使用新 MAC。
5. 验证：
   - `sudo -n hcitool con` 不要求密码；
   - `chronyc tracking` 显示已同步；
   - `systemctl is-active ble-time-bridge` 返回 `active`；
   - 两件衣服的时钟报告不再为零，运行状态显示 `utc=True`。

建议安装入口（执行前单独复核依赖与蓝牙双角色能力）：

```bash
cd /agibot/data/home/agi/x2_shadow_20260814_1411/x2_robot_teleop_20260814/ble-time-bridge
sudo env FGP_PYTHON=/usr/bin/python3 ./bridge.py install
sudo systemctl status ble-time-bridge --no-pager
```

## 1. 每次开机：官方机器人基线

1. 等待机器人系统初始化完成。
2. 用官方手柄 `L2+X` 进入 `JOINT_DEFAULT`。
3. 双脚完全着地、有人保护时，用 `R2+X` 进入 `STAND_DEFAULT`。
4. 从 PC2 只读确认模式为 `STAND_DEFAULT`、状态 100。
5. 录制静止基线或至少确认 IMU、关节 state/command 正常。

## 2. 每次穿衣：BLE 前置检查

1. 上衣和裤子上电，手机 nRF Connect 不保持连接。
2. 分别做 15–30 秒单设备测试。
3. 打开自动低延迟参数，再做双设备测试：

```bash
export BNO085_BLE_AUTO_LECUP=1
export BNO085_BLE_LECUP_MIN=8
export BNO085_BLE_LECUP_MAX=12
export BNO085_BLE_LECUP_LATENCY=0
export BNO085_BLE_LECUP_TIMEOUT=500
```

4. 双衣验收门槛：
   - 上衣、裤子都持续约 `60–70 Hz`，不能一件长期停在 40 Hz；
   - 两件衣服均 `utc=True`；
   - 配对时间差不超过 `40 ms`；
   - host age 不超过 `150 ms`；
   - 断连、积压补发和持续速率塌陷均为失败。

低于门槛时不做 T-Pose，先解决 BLE/时钟问题。

## 3. P3 Shadow：穿衣运行但绝不控制机器人

机器人保持官方 `STAND_DEFAULT`。启动参数必须同时包含：

```bash
cd /agibot/data/home/agi/x2_shadow_20260814_1411/x2_robot_teleop_20260814/x2_pc1_ble_teleop

BNO085_BLE_AUTO_LECUP=1 \
PYTHON_BIN=/agibot/data/home/agi/miniconda3/envs/teleop/bin/python \
TELEOP_TERMINAL_MODE=debug \
./run_pc1_ble.sh \
  --teleop-backend none \
  --shadow-log /agibot/data/home/agi/data/x2_shadow_logs/session.jsonl
```

只有终端同时满足以下条件才按回车：

- 上、下衣 buffer 均准备完成；
- `utc=True`；
- 终端显示 shadow logging，且 UDP publish rate 为 0；
- 操作者已经摆好标准 T-Pose。

校准后依次采集：静止、双臂前举、双臂侧举、躯干小幅转动、小幅屈膝。机器人不跟随。离线检查输出延迟、跳变、关节限位、左右对称、衣服目标与官方稳定站立目标的冲突。

## 4. 真机接管门禁

只有 P3 通过后才允许真机实验。当前代码会生成完整 62 维目标，因此在确认下肢控制契约前，不直接沿用旧流程按 `R2+O`。

旧流程的 `aima em stop-app teleop_bridge` 也不能直接照搬：新代码的 official backend 正是向 `10.0.1.40:50040` 发送 AIMRT UDP，并依赖机器人侧桥接链。Shadow 阶段绝不停止它；真机阶段必须先验证“UDP 接收器—topic publisher—MC subscriber”的实际职责，再决定是否有需要停止的冲突模块。

真机第一阶段应采用：

1. 官方稳定站立保留下肢平衡权；
2. 衣服仅获得限幅、限速的上半身控制权；
3. 官方手柄保持最高优先级安全接管；
4. `R2+X` 回到稳定站立作为首要恢复动作；
5. 单膝跪、后仰及大幅屈膝继续禁用。

## 5. 结束与关机

1. 先停止衣服进程，确认 UDP/ROS 发布为零。
2. 用手柄回到 `R2+X` 稳定站立。
3. 按官方流程 `L2+X` 进入站姿预备。
4. 在人工支撑下进入阻尼/零力矩，再关机；不得让直立机器人直接进入零力矩。
