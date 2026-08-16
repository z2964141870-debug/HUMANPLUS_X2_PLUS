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
