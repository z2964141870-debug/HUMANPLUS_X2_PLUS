# BASE Phase40：Phase34 原生稳定行走盆地对照

日期：2026-08-11
状态：**BOTH BRIDGE ENDPOINTS OUTSIDE COHERENT STAGE250 BASIN / ZERO RUN**

## 目的与合同

用户已提供过稳定迈步数据。本阶段直接复用 Phase34 full-gate closed AimDK straight trace，不重新采集。700个安全 telemetry rows 中，650个同时具有后续50行（约1秒）安全未来，作为 coherent anchors。

可比字段为 q31、dq31、当前 issued action15、projected gravity、root z/tilt。每组阈值由 Phase34 自身 leave-one-out 最近邻 p95 标定；目标必须来自同一行，禁止把不同时间的各组最近邻拼接。root XY/yaw 因任务进度和全局坐标不同而排除。

## 结果

### Phase26 CEM best endpoint

- coherent minimax distance：`35.70×`。
- q31：`35.70×`，原始 L2约`2.60 rad`。
- issued action：`4.43×`。
- gravity：`22.27×`。
- root posture：`30.24×`。
- 最近 coherent anchor 是 Phase34 stop `6.98s`，但仍远未入域。

### Phase39 sequence-guided endpoint

- coherent minimax distance：`38.18×`。
- q31：`37.13×`。
- dq31：`4.35×`。
- issued action：`2.17×`。
- gravity：`38.18×`。
- root posture：`36.07×`。

## 结论

Phase34 稳定行走确实提供了成功动力学盆地，但 Phase26/39 的 stop/recovery endpoint 与它不是“小幅 bridge”关系：q、gravity、root posture 同时相差几十个 Phase34 内部 LOO-p95。Phase39复制 recovery success suffix 也没有向 Stage250 原生盆地靠近。

因此不能把 Phase34 当作直接末端 target，再用现有1秒小 residual bridge硬接过去。它更适合作为独立 locomotion backend 或 keyframe skeleton 的起始/结束稳定片段；中间必须重新规划足位、COM和contact transition。该结论为只读距离审计，不证明两策略之间不存在更长的闭环路径。

当前裁决：`NATIVE_WALK_BASIN_EXISTS / CURRENT_STOP_ENDPOINTS_NOT_LOCALLY_CONNECTABLE`。0 physics、0训练，训练仍锁。
