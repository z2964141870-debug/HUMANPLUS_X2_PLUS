# X2-Sonic 官方复现与衣服日志离线边界（2026-08-16）

## 当前结论

3090 上已经重新复现官方 X2-Sonic ONNX + 官方 MuJoCo 场景。模型以
CUDAExecutionProvider 加载，输入为 1670 维、输出为 31 维，50 Hz 控制和
每 tick 4 个 MuJoCo substep 均正常。原始 PHUMA 输入的闭环结果仍然不稳定：
这不是“权重无法加载”，而是 motion distribution 与 X2 动力学闭环的稳定域
不一致。

本次小规模复现报告：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-16/official_sonic_x2_repro_smoke8_5s.json
~~~

命令：

~~~bash
cd /home/yu/projects/BFM-Zero
ENVROOT=/home/yu/miniconda3/envs/x2-sonic-isaaclab
CUDA_LIBS=$(find "$ENVROOT/lib/python3.11/site-packages/nvidia" -type d -name lib -print | paste -sd: -)
LD_LIBRARY_PATH="$CUDA_LIBS" /home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python +  tools/official_x2/eval_official_sonic_x2.py +  --model /media/yu/FAFF-E977/data/BFM-Zero/raw/sonic-x2/x2_sonic_policy.onnx +  --scene /home/yu/projects/sonic-web-demo-x2/assets/robot/scene.xml +  --motion-root /home/yu/projects/x2_teleop_final/x2_sonic/motion_lib_x2 +  --clips-per-set 2 --max-clips 8 --seconds 5 --require-cuda +  --output /media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-16/official_sonic_x2_repro_smoke8_5s.json
~~~

这 8 条记录中，4 条完成 5 秒，4 条分别在约 1.08 秒或 1.88 秒触发
root height/tilt safety gate。不同 PHUMA 子集之间存在相同 source SHA 的重复，
因此这次是 smoke replay，不应被当成 8 个独立动作的统计结论。完整的 96 条
分层结果和 30 秒结果仍以 2026-08-15 的报告为准。

## 已固化的离线适配器

### 实时逐帧适配器

代码：

~~~text
/home/yu/projects/BFM-Zero/tools/official_x2/x2_sonic_input_adapter.py
~~~

当前仅包含已经在 MuJoCo 中验证过的输入域干预：

- root tilt scale = 0.0（保留 yaw，压掉 roll/pitch）
- pose scale = 0.5（围绕官方站立角缩放 31 维关节目标）
- 可选的因果 joint speed limit

它只做 NumPy 计算，不导入 MuJoCo、ONNX、BLE、ROS，也不连接机器人。

### 衣服日志 JSONL 回放边界

代码：

~~~text
/home/yu/projects/BFM-Zero/tools/official_x2/replay_x2_sonic_adapter.py
~~~

输入暂定为一行一个 JSON 对象：

~~~json
{"timestamp_s": 0.0, "fps": 50.0, "joint_pos": [31个弧度值], "root_quat": [w, x, y, z]}
~~~

约束：

1. timestamp_s 必须有限且单调不减；
2. joint_pos 必须是 31 维有限值；
3. root_quat 使用 scalar-first [w,x,y,z]，不能是零四元数；
4. fps 可逐帧提供；缺省时使用 50 Hz；
5. 输出仍是 JSONL，并且每帧附带 raw/adapted root tilt、是否触发关节限速
   等 telemetry；
6. 同时生成 manifest，记录输入/输出 SHA256、适配参数和硬件边界；
7. garment_mapping 状态目前明确标为 pending，不把 BLE 私有协议猜成结论。

测试：

~~~bash
cd /home/yu/projects/BFM-Zero
PYTHONPATH=/home/yu/projects/BFM-Zero/tools/official_x2 +  /home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python -m unittest -v +  tools/official_x2/test_x2_sonic_input_adapter.py +  tools/official_x2/test_replay_x2_sonic_adapter.py
~~~

当前结果：5 tests passed。

另外使用已有 PHUMA canonical motion 的前 120 帧做了一次真实数据格式回放，
不是合成零输入：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-16/canonical_replay_smoke/
~~~

逐帧 replay 输出与同一动作的 batch root_tilt=0, pose=0.5 artifact 比较，
120 帧的 joint 和 quaternion 最大绝对误差均为 0.0。对应 parity manifest：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/canonical_replay_smoke_root0_pose05_parity.json
~~~

本次复现和回放的总 manifest：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/x2_sonic_repro_manifest_v2.json
~~~

## JSONL → observation/action parity

新增工具：

~~~text
/home/yu/projects/BFM-Zero/tools/official_x2/parity_x2_sonic_jsonl.py
~~~

它直接读取衣服回放边界的 JSONL，不依赖 PHUMA pkl。为了把“字段/单位/顺序
错误”和“动力学不稳定”分开，当前使用确定性的 synthetic qpos/qvel trace，
并明确将其标记为非动力学证据。100 个 policy ticks 的结果：

~~~text
observation: shape (100,1670), max_abs=0.0, SHA identical
action:      shape (100,31),   max_abs=0.0, SHA identical
provider:    CUDAExecutionProvider + CPUExecutionProvider
~~~

报告：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/canonical_replay_smoke_observation_action_parity.json
~~~

这证明 canonical JSONL 经实时 adapter 后可以无损进入官方 tokenizer/history/ONNX
接口；下一步只需把 synthetic state 替换为 MuJoCo replay state，才能进入真正的
闭环稳定性判断。

## 真实 MuJoCo state parity 与 matched closed-loop 对照

新增 state trace：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/state_trace_rottilt0_pose05_100ticks.json
~~~

该 trace 记录官方 MuJoCo 闭环每个 50 Hz tick 交给 policy 前的 qpos/qvel，
100 ticks 全部采集完成，没有 root height/tilt safety gate。

在这条真实 state trace 下，batch reference 与实时 JSONL replay 的
observation/action 最大误差仍为 0.0。然后对同一条 matched motion 进行官方
闭环对照：

| 输入 | 生存时间 | loop reset | 结果 |
|---|---:|---:|---|
| raw canonical | 1.88 s | 0 | root height/tilt gate |
| root_tilt=0 + pose_scale=0.5 | 30.0 s | 5 | 完成，无跌倒 |

对照报告：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-16/official_raw_vs_adapted_smoke_comparison.md
~~~

这是一条 matched-motion 的因果性 smoke evidence，不是对所有 PHUMA 或衣服
动作的普适成功声明；总体结论仍以 96 条分层、30 秒实验为准。

## pose-scale 稳定边界矩阵（96 条分层动作）

为了避免把 `pose_scale=0.5` 的单点结果误认为最终方案，在相同官方 ONNX、
官方 MuJoCo scene、`root_tilt_scale=0` 和相同 96 条 PHUMA 分层动作下，
只扫描 pose scale。每个档位运行 5 秒，严格要求所有选中 clip 通过官方
root-height/tilt safety gate：

| pose scale | 通过/总数 | 最短生存 | 最大漂移 | 最大倾角 | 结论 |
|---:|---:|---:|---:|---:|---|
| 0.50 | 96/96 | 5.00 s | 1.5937 m | 0.1038 rad | 通过 |
| 0.60 | 96/96 | 5.00 s | 2.1842 m | 0.1445 rad | 通过 |
| 0.65 | 94/96 | 2.32 s | 2.6243 m | 0.1506 rad | 首次失败 |
| 0.70 | 94/96 | 1.50 s | 3.0719 m | 0.1594 rad | 更早失败 |
| 0.75 | 94/96 | 1.28 s | 3.6937 m | 0.4396 rad | 更早且更剧烈 |

`0.65/0.70/0.75` 的失败都来自同一动作族
`animation__Ways_to_Stand_Downhill_Skateboarding_clip1_chunk_0000`；由于
96 条选择保留多个 PHUMA 子集，该内容出现两次，并不代表两个独立动作。
这给出一个当前闭环的可复现稳定边界：在这个分层 5 秒测试中，0.60 仍在
安全域，0.65 开始出现失稳。它只说明当前权重/动力学契约的稳定域，不能推出
动作表达性、衣服噪声鲁棒性或真机安全性。

矩阵 manifest：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/x2_sonic_pose_boundary_matrix_v1.json
~~~

可读报告：

~~~text
/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-16/x2_sonic_pose_boundary_matrix.md
~~~

## 下一阶段计划

### A. 衣服日志接入（不涉及硬件控制）

拿到一段衣服原始日志后，只做字段盘点和离线转换：

1. 找出时间戳、上衣/裤子帧号、IMU 四元数或旋转表示、校准状态；
2. 明确衣服关节/骨骼到 31 个 X2 canonical joints 的映射；
3. 生成上述 JSONL，不修改原始日志；
4. 用 replay 工具检查单调性、丢帧、延迟、有限性和适配前后统计；
5. 用相同输入跑 batch canonical adapter，逐帧比较输出，要求误差为数值精度量级。

### B. observation parity

将回放后的 canonical frame 接入官方 observation tokenizer，只比较：

- observation 维度和历史堆叠顺序；
- 首帧和固定窗口的 SHA256；
- ONNX action 的有限性、幅值和时间连续性；
- 与官方 PHUMA replay 的差异。

这一步不能证明真机安全，但能先排除“衣服字段/顺序/单位错了”。

### C. 分布与动力学实验

沿用已经完成的单变量矩阵，不直接训练新 policy：

1. raw；
2. root tilt scale；
3. pose scale；
4. joint speed limit；
5. 后续才考虑 phase/contact-aware gate。

每次只改一个变量，保留原始数据、输出、报告和 SHA256。当前 96 条分层实验
中，root tilt=0 + pose=0.5 是稳定性探针，不应直接宣称为最终 expressive
teleoperation policy。

### D. 机器人前置条件

只有 A/B/C 在离线数据上通过后，才整理真机实验清单：停止旧 bridge、确认
控制模式、低速/限幅、急停和回退路径。当前目标期间不操作 Orin、BLE 或真实 X2。

## 版本与存储

- 项目根目录：/home/yu/projects/BFM-Zero
- 代码写入：/home/yu/projects/BFM-Zero/tools/official_x2
- 实验数据：/media/yu/FAFF-E977/data/BFM-Zero
- 当前分支：work/codex-x2-extremity-contract
- 本报告提交时的 commit 由 Git manifest 记录；工作树中的既有 artifacts/
  和 external/humenv/ 未删除、未加入本次提交。
- 原始模型与 PHUMA 数据只读；日志、权重和报告不进入 Git。
