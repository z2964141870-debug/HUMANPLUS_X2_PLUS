# X2 WBT Canonical Generation Contract Phase6

- 裁决：**PHASE6_CANONICAL_SOURCE_TIME_STOPPED_NOT_PROMOTABLE**。
- 无训练、无 teacher 搜索、无 checkpoint、无真机；只做固定五动作零更新 reference/PD 对照。
- official MuJoCo collision/contact、COM、DCM 均是模型估计量，**不是实机 GRF/COP/足底力真值**。

## 假设与单变量

Phase5 turn/stop 源段没有静止拼接点。A 只把 source-time 改为：0.5 s 静止前缀 → C2 ramp → 全程 0.4× source speed；关节、root、ground、contact、PD、控制频率均不改。若任一动作 survival 下降 >0.1 s 或 slip 越门，立即停止，不运行 B（root-ground/contact）或 turn teacher。

## 连续性、turn 与官方物理门

| role | continuity | join q/root-lin/root-ang jump | turn root/L-foot/R-foot yaw | prescribed full | Phase5→A free(s) | Δ(s) | slip/limit | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walk_straight | True | 0.035/0.001/0.004 | - | 1.000 | 1.808→1.879 | +0.071 | 0.0073/0.0281 | True |
| turn_left | True | 0.146/0.080/0.486 | 2.474/2.433/2.577 | 1.000 | 1.573→1.560 | -0.013 | 0.0149/0.0446 | True |
| turn_right | True | 0.101/0.043/0.218 | -0.830/-0.936/-0.875 | 1.000 | 1.134→1.135 | +0.001 | 0.0066/0.0276 | True |
| stand_to_walk | True | 0.053/0.003/0.033 | - | 1.000 | 1.605→1.621 | +0.016 | 0.0031/0.0230 | True |
| walk_to_stand | True | 0.124/0.025/0.081 | - | 1.000 | 2.665→2.442 | -0.223 | 0.0136/0.1619 | False |

## B1 canonical / Phase5 ground 前置门

- official 31DoF 顺序一致且 head 2DoF 锁零：`True`。WBT 控制边界仍是 29DoF；缓存保留 31 列仅用于匹配 MJCF，最后两列不参与动作。
- Phase5 已有 root-ground 离线门：`False`；失败动作：`['turn_left', 'turn_right', 'stand_to_walk', 'walk_to_stand']`。Phase6 没有把旧失败 B 冒充新 canonical contract。

## 停止裁决

- 结果：source-time A 在 ['walk_to_stand'] 触发 survival/slip 停止门；未执行 B。
- 结论：0.4× source-time 虽修复五动作连续性，但不能跨五动作保持 free-root 生存；只否定该生成器。
- 下一步：停止扩展该分支；回到原始 source 生成器，优先修数据级 reset/ground/contact 联合边界，而非继续扫时间尺度。
- 本结果只否定当前 0.4× source-time 生成器，不能据此否定 X2 动力学、Any2Any 或 WBT 路线。
