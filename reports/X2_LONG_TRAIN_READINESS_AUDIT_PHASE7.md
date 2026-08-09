# X2 双主线长训准备度审计：Phase10 更新

日期：2026-08-09

证据截止：Git `6337791` 后的 Phase7–10 工作树，AimDK v1.0 official MuJoCo

裁决：`LONG_TRAIN_LOCKED_BOTH_TRACKS`

## 目的

本表按任务卡逐项证明“现在可以做什么、还缺什么”，防止用局部视频、训练 reward 或单条动作替代长训解锁条件。未列为通过的项目均视为未完成。

## 主线 A：速度型保底后端

| 任务卡要求 | 当前证据 | 状态 |
|---|---|---|
| 冻结 BASE_LOCOMOTION / BASE_TRANSITION 资产与 contract | `x2_speed_backend_manifest.*`；checkpoint/ONNX/template/PD/obs/action/hash 已记录 | 通过 |
| nominal 基础能力不低于 Stage250 24/24 | 历史 Stage250 24/24；但其视觉审计发现转向仅约 ±18°、持续后仰约 10–11° | 位移门通过，质量门不完整 |
| stop-to-stand/recovery 的真实状态初始化 | 90 条 Stage335 状态已入库；Phase6 修复首轮 dq 被清零，投影后物理 snapshot 精确 | 基础设施部分通过 |
| physical + controller deterministic suffix | Phase8 在未修改 official MJCF 的 test-only 同进程分叉中，snapshot 与 10/25/50 tick q/dq/root/action/event/pitch 全部 exact；但 closed AimDK/ROS 无 full-state injection，尚不能证明 ROS suffix | direct-MJCF 通过，closed ROS 未证明 |
| stiff-fixed 至少 5/5 | 冻结 Stage306 为 3/5；后续分支均未形成可靠 5/5 | 未通过 |
| stiff-fast 至少 5/5 | 历史 Stage326 5/5，但不能替代 fixed 或完整矩阵 | 局部通过 |
| 同候选 PD×upper×motion×seed 完整矩阵 | 尚无相同 checkpoint/配置的完整 81-case | 未通过 |
| signed pitch / natural posture | Phase7 分阶段分析器已建立；默认姿态只改善 0.84°、单独缩腰完整运动只改善 0.016°，主因收敛到全下肢 sagittal target×PD/default/controller 联合平衡点 | 根因收敛，质量门仍未通过 |
| frozen actor 左右等变性修复 | Stage355 证明漂移符号翻转；Stage356 部署时 roll/yaw 群平均导致反向失稳倒地 | 该修法否决 |
| 5-update stateful recovery RSI | Phase9 同 source/seed 对照：source/f000/f005 均 full `0/5`；f005 将 startup 由 f000 `0/5` 恢复到 `5/5`，heading 0.628→0.428 rad、横漂 0.551→0.410 m，但 move/stop 仍 `0/5`，stop drift 0.423→0.452 m且后仰退化 | 局部信号，不晋级，25-update 锁定 |

BASE 长训阻断项：

1. deterministic suffix 的 state sufficiency 已在 official MJCF test-only runner 证明，但 closed AimDK/ROS 没有中间态注入接口；
2. stiff-fixed 未达 5/5；Phase9 证明少量 stateful RSI 能修复起步但不能统一修复 move/stop；
3. signed pitch 已可复现，但尚无官方 locomotion 自然阈值，也没有修复候选；
4. 统一完整矩阵未运行。

## 主线 B：Whole-Body Tracking

| 任务卡要求 | 当前证据 | 状态 |
|---|---|---|
| 官方 31DoF / WBT 29DoF / BASE 15DoF canonical mapping | official model/joint/body/mirror contract 与 FK/mirror tests | 通过 |
| 20–30 条诊断面板 | 固定 24 条，source 与 SHA 已核验 | 通过 |
| 3 条旧/官方 MJCF A/B | 数值完全相同，否定“只换模型文件即可修好 reference” | 通过（否证） |
| Bronze/Silver/Gold 自动门 | tier gate 与 official prescribed/free runner 已建立 | 工具通过 |
| canonical reset 起点 | Phase5 静止前缀改善部分动作，但 turn join/stop 仍不统一 | 未通过 |
| canonical robot-level root-ground | Phase7 排除 visual mesh，只用官方 24 个 active sole spheres，并用唯一 MA9 公式消除 source absolute pelvis-height 泄漏；五动作 geometry/prescribed/survival 全过 | Bronze 几何通过 |
| canonical contact realization | 五动作 intent-vs-official collision agreement 仅 0.362–0.633；walk_to_stand slip 0.0671→0.1603 m/s，超过 0.0871 门 | 未通过 |
| stance-foot XY lock | Phase8 的碎片接触窗首帧 anchor+C2+局部 IK 使 5/5 滑移恶化、4/5 joint-step 越门，离线即停止 | 该生成器否决 |
| 五动作 dynamic teacher | 最好仅单条 walk existence smoke；最终面板 1/5，2.456 s 后仍倒 | 未通过 |
| dynamic Silver / Gold 数据 | GMR/AMASS/PHUMA/BONES dynamic Silver 仍为 0；Phase10 新增 1 份 60 s SDK-native Gold sanity seed，train 32 s / embargo 4 s / held-out 24 s | 原生 Gold 管线种子通过；跨具身数据门仍未通过 |
| official native Gold qualification | 3000 帧/59.98 s，29 项门全过；active-sole DS/SS/flight=0.807/0.192/0.001，44 个 DS→SS→DS 周期 | 可用于 pipeline sanity，不是 GMR/AMASS Silver，不证明 Any2Any |
| native Gold MotionLib ingestion | Phase11 held-out 3×400@50 Hz；q max 2.38e-7 rad、root pos exact、quat max 4.30e-7 rad、official FK pos/ori max 3.11e-7 m / 4.58e-7 rad；WBT29/head-lock 通过 | 通过；actual dq/root velocity/contact 须显式 state adapter 恢复 |
| faithful Any2Any 0/5/25 smoke | Phase10 Gold 已通过 MotionLib ingestion/round-trip；下一步仅允许 held-out clip0 冻结 prescribed/free reference-trackability smoke；跨具身训练仍要独立 Silver/held-out | Phase12 无训练物理 smoke 可解锁，PPO 仍锁定 |
| held-out WBT 动作门 | 没有可训练动态集，因此尚无有效 baseline 候选 | 未启动 |

WBT 长训阻断项：

1. robot-level root-ground Bronze 已统一，但 frame0/reset 动态连续性与 contact realization 尚未统一；
2. walk_to_stand stance slip 和五动作 contact agreement 未过门；
3. 已有 SDK-native Gold sanity seed，但 GMR dynamic Silver 仍为 0；在 MotionLib ingestion 与 zero-update 通过前不启动 PPO，且原生 Gold 不能替代跨具身 train/held-out 证据。

## 当前允许的下一动作

### BASE

1. 保留 Phase7 signed pitch 分阶段指标，未来候选必须相对 Stage250 改善且不能牺牲 survival/速度；
2. direct official-MJCF stateful reset 可进入 Isaac recovery 训练的接口验证，但不能改写为 closed ROS 证据；
3. 若要完成 closed ROS suffix，只能等待厂商暴露 full-state injection 或建立明确的 simulator test hook；
4. Phase9 5-update smoke 已否决，不运行 25-update；下一个 BASE 干预必须把 recovery 学习与已通过的 stand/start 保护明确分开，不再只调 RSI 比例。

### WBT

1. 冻结 Phase7 机器人级 root-ground Bronze 公式，不再调 clearance/window；
2. 固定 hysteresis/min-dwell contact schedule 已在 Phase9 WBT 否决，不再扫描该类参数；
3. Phase11 已证明 Phase10 SDK-native Gold 可被现有 MotionLib 无损读取；下一步只对 held-out clip0 运行一次冻结 prescribed/free reference-trackability smoke，用来裁决管线而非跨具身能力；
4. 原生 Gold pipeline sanity 通过后，仍需形成至少一组 GMR Silver/Gold train/held-out，才启动 faithful Any2Any 5-update。

## 明确禁止

- 不因 Stage250 会前进就启动 BASE 长训；
- 不因单条 walk teacher 延长生存就命名为 Silver；
- 不继续扫 frozen actor symmetry alpha/mask；
- 不继续扫 foot offset、时间尺度或小 root 偏置；
- 不把模型 contact/COM/DCM 写成实机 GRF/COP 真值；
- 不向真机发命令。

## 总结

官方物理域带来的有效推进是：已经把 BASE 失败缩小到“recovery RSI 能修起步，但会与 move/stop/姿态互换”，把 WBT 分成“管线是否能学 X2 原生 Gold”与“GMR reference 是否可执行”两个可独立证伪的问题。两条线仍未解锁长训，但不再是一个混在一起的“动力学矛盾”。
