# X2 Phase10：动态重定向 actuator/head 映射纠正

日期：2026-08-11

状态：**CONTRACT BUG CONFIRMED / PHASE6-8 MECHANISM SURVIVES ZERO-SEARCH PROJECTION / NEW SEARCHES STILL LOCKED**

资源：4 次 deterministic raw replay；单线程、`nice=10`、0 GPU；新搜索 0；RL/PPO 0。

## 1. 问题

官方 actuator 顺序中：

- `head_yaw_joint/head_pitch_joint` 是索引 `15/16`；
- 最后两个 actuator `29/30` 是 `right_wrist_pitch_joint/right_wrist_roll_joint`。

Phase4/5/6/8 的部分代码使用了 `[:-2]`、`[-2:]`，把“最后两个 actuator”误当 head。这导致：

- Phase4/5/8 的搜索空间不是声明的 body29；它包含真实 head，遗漏两个右腕关节；
- Phase6/8 nominal target 把两个右腕设为零，而没有按名字固定 head；
- Phase3b/3c 的优化器本身按名字固定 head，正确；但 raw replay 的 `head_q_max` 指标实际测量了右腕。

因此在此纠正前，不允许启动 Phase10 事件式搜索。

## 2. 零搜索纠正方法

没有重跑任何优化器。确定性构造正确映射：

1. body29 = official 31 actuators 排除名字为 head 的两项；
2. head target 在每个 control tick 显式置零；
3. Phase6 nominal bridge 中，右腕恢复原 reference/bridge target，不再错误置零；
4. Phase8 已保存的 5×29 correction 仍按其历史列语义读取；丢弃真实 head 两列的 correction，历史未包含的两个右腕 correction 置零；
5. 在未改写 official raw torque-motor model 中重新播放。

这只是历史 candidate 的合同投影与归因审计，不是重新优化后的候选。

## 3. 结果

| 指标 | 历史报告 | 正确映射复核 |
|---|---:|---:|
| Phase6 baseline fall | 0.683 s | **0.683 s** |
| Phase6 left/right switches | 0 / 0 | **0 / 0** |
| Phase6 head max | 旧指标实际是右腕 | **0.000776 rad** |
| Phase8 candidate 0.70 s | full | **full** |
| Phase8 full continuation fall | 0.775 s | **0.775 s** |
| Phase8 left switches | 26 | **26** |
| Phase8 longest left-off/right-on | 21 ms | **21 ms** |
| Phase8 left non-intent contact | 79.19% | **79.19%** |
| Phase8 corrected head max | 未可信 | **0.000589 rad** |
| Phase8 right wrist max | 历史被当作head | **0.004490 rad** |

Phase6 corrected zero replay两次逐字段完全一致。

Phase8 纠正投影后的 stance speed p95 为左/右 `0.1023/0.1014 m/s`，右支撑 excursion `0.0781 m`；仍然不是合格 teacher。核心的“延寿 + 接触 chatter 而非稳定 liftoff”解释不变。

## 4. 裁决

1. **Phase4/5 的方法裁决进一步降级。** 它们的搜索空间本身不满足声明的 body29/head2 合同；既有 candidate 可以作为失败证据，但不能据此评价忠实 SBTO/DDR。
2. **Phase6 reset projection 保留。** 投影变量本来就由 joint name 构造，不受 actuator 尾部假设影响；正确 target mapping 下仍确定性跌倒于 0.683 s。
3. **Phase8 的局部正信号保留但身份更正。** 历史搜索合同有误；不过对保存 candidate 做正确映射投影后，0.775 s、26次切换、21 ms最长离地逐值保留。因此可以继续研究该机制，但新 runner 必须按名字建立 body29/head2。
4. **Phase9 不受该具体错误影响。** 它按名字找到 head 索引并冻结 bounds；其长窗口 NLP/raw replay失败裁决不变。
5. **新搜索仍锁。** 下一步先实现 name-based runner、body29 roundtrip、head2 nominal和zero-candidate exact门；通过后才能运行唯一低资源事件式 sampling。

## 5. 产物

- `phase10_actuator_mapping_audit.json`
- `audit_x2_retarget_actuator_mapping_phase10.py`
- 本报告
