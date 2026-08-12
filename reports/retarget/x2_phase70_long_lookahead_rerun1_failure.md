# Phase70 v4 repair rerun：技术无效并停止

权威实验裁决为 `FAIL_INVALID_STOP`；治理状态为 `FAIL_IMPLEMENTATION_STOP_NO_MORE_PHASE70_RERUN`。v4 已消耗 Phase70 允许的最后一次物理启动；禁止第三次 Phase70 启动。

本次完成了 `64 env × 400 steps`，全部 25,600 个 transition 有效，0 termination/timeout，峰值显存 3269 MiB，磁盘增量 158,871,552 bytes。模型、residual、optimizer 均未改变：optimizer step 为 0、optimizer state 为 0、checkpoint 为 0。

三项技术门失败：CPU/CUDA RNG 的预注册值采用 raw-state bytes SHA，而 runner 使用带 dtype/shape 前缀的 `tensor_hash`；additive gradient 的独立 float32 递推 relative-L2 为 `1.39548e-5`，超过预注册 `1e-5`，尽管 cosine 为 `0.999999999914`。历史门禁不追溯放宽。

raw bundle 已安全落盘，但只能用于标明 `posthoc/no-promotion` 的离线校准。探索性数值显示 pitch reward 自身与正 pitch 同向，而 primary total 与正 pitch 反向；同时两个 env cohort 方向 cosine 为 `-0.3397`，support 方向区间跨零，因此不能据此训练或改判通过。

Phase70 到此永久关闭。下一步最多是零物理、零优化的现有 bundle 数值审计，以及全新预注册的问题定义；长训、导出、部署和 Task2 完成都未解锁。
