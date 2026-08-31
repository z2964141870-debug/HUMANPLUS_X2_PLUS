# X2 ZMQ v5.1 Wire 格式权威规格(衣服参考通路)

更新: 2026-08-29 · 适用于 `--strict-reference-velocity` 打开的实时衣服参考通路

## 0. 这份文档的地位与事实来源

本文**不是**设计意图的复述,而是从机器人侧 C++ 解码器**逐行提取**出来的实际行为。
每条结论都标了来源行号,发送端作者应当把行号当作复核入口,而不是相信本文的措辞。

事实来源(按权威性排序):

| 优先级 | 文件 | 作用 |
| --- | --- | --- |
| 1 | `src/x2/agi_x2_deploy_onnx_ref/src/zmq_pose_input_source.cpp` | 字段名匹配、dtype/长度校验、strict 拒帧、速度转发 |
| 1 | `src/x2/agi_x2_deploy_onnx_ref/include/zmq/zmq_packed_message_subscriber.hpp` | topic 前缀、1280 字节 header、分帧与 offset 走位 |
| 1 | `src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp` | `NUM_DOFS` / `NUM_FUTURE_FRAMES` / `DT_FUTURE_REF` / `default_angles` |
| 2 | `src/x2/agi_x2_deploy_onnx_ref/include/zmq/zmq_pose_input_source.hpp` | 语义注释(注意:第 50 行的 `frame_index_future` 在 .cpp 里**并未解析**) |
| 3 | `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py` | Python 侧参考打包实现(header 压缩分隔符 + NUL 补齐) |
| 3 | `docs/source/references/x2_zmq_protocol.md` | 跨语言协议总览(部分条目与 X2 解码器实际行为不符,见第 9 节) |

配套参考实现:

* 发送端:`scripts/garment_zmq/publish_v51_reference.py`
* 纯 Python 自校验解码器:`scripts/garment_zmq/verify_v51_bytes.py`

## 1. 常量

| 常量 | 值 | 来源 |
| --- | --- | --- |
| `NUM_DOFS` | **31** | `policy_parameters.hpp:54` |
| `NUM_FUTURE_FRAMES` | **10** | `policy_parameters.hpp:56` |
| `kFutureSlots`(严格未来 slot 数) | **9** = `NUM_FUTURE_FRAMES - 1` | `zmq_pose_input_source.cpp:244` |
| `DT_FUTURE_REF`(未来 slot 间距默认) | **0.1 s** | `policy_parameters.hpp:60` |
| `CONTROL_DT`(策略周期) | 0.02 s(50 Hz) | `policy_parameters.hpp:57` |
| `DEFAULT_HAND_DOF_PER_SIDE` | **10**(硬编码,不随 `--hand-dof` 变) | `zmq_pose_input_source.hpp:156` |
| `HEADER_SIZE` | **1280** 字节 | `zmq_packed_message_subscriber.hpp:99` |
| 默认 host/port/topic | `127.0.0.1` / `5556` / `pose` | `run_x2_suspended_sonic.sh:197-201` |

速度由发送端显式提供,C++ 不重估。`fixed` 模式用 **50 Hz** 保持 MuJoCo
LiveMotion parity;实时 HMCP 用协议发送时间求真实 `dt`(见第 6、9.1 节)。

## 2. 消息封装(单个 ZMQ part)

```
[ topic_bytes ][ 1280 字节 JSON header(NUL 右补齐) ][ 各字段二进制,按 header 顺序拼接 ]
```

* **一个 ZMQ part,不是多 part**。`PollOnce` 只 `recv` 一次
  (`zmq_packed_message_subscriber.hpp:265-267`)。用 `socket.send(msg)` 一次发完。
* **topic 前缀是裸字节拼在最前面**,不是 ZMQ 多帧 envelope。订阅端用
  `memcmp` 比对前缀并跳过(`:273-280`)。topic 必须是 `pose` 的 UTF-8 字节。
* header 长度必须**正好** 1280 字节。剥掉 topic 后 `packed_size < 1280` 整包丢弃
  (`:282-288`)。
* header 用 `strnlen(..., 1280)` 取长度(`:292-293`),所以 JSON 之后必须用
  `\x00` 补齐,且 JSON 本身**不能含 NUL**。JSON 超过 1280 字节无法表达
  (Python 侧直接抛异常,`zmq_planner_sender.py:25-26`)。
* 二进制段**无对齐填充**:字段按 header 里 `fields` 数组的顺序紧密拼接,
  offset 由 `ComputeByteSize()` 累加(`:334-346`)。
* 订阅端设置了 `conflate=1` + `rcv_hwm=1`(`zmq_pose_input_source.cpp:89-92`),
  所以只有最新一帧会被处理,积压帧被丢弃。发送端不必做流控,但也不能指望
  历史帧被消费。

### 2.1 实际尺寸(本规格的最小必需字段集)

| 段 | 字节 |
| --- | --- |
| `b"pose"` | 4 |
| JSON header | 1280 |
| 二进制载荷 | 2652 |
| **合计** | **3936** |

载荷明细:`joint_pos_mj` 124 + `joint_vel_mj` 124 + `root_quat_xyzw` 16 +
`joint_pos_mj_future` 1116 + `joint_vel_mj_future` 1116 +
`root_quat_xyzw_future` 144 + `future_dt_s` 4 + `frame_index` 8。

压缩分隔符下 JSON header 实测 475 字节(最小集)/ 693 字节(加上双手 + token +
`frame_index_future`),距 1280 上限有余量。

## 3. JSON header schema

```json
{"v":5,"endian":"le","count":1,"fields":[
  {"name":"joint_pos_mj","dtype":"f32","shape":[31]},
  {"name":"joint_vel_mj","dtype":"f32","shape":[31]},
  {"name":"root_quat_xyzw","dtype":"f32","shape":[4]},
  {"name":"joint_pos_mj_future","dtype":"f32","shape":[9,31]},
  {"name":"joint_vel_mj_future","dtype":"f32","shape":[9,31]},
  {"name":"root_quat_xyzw_future","dtype":"f32","shape":[9,4]},
  {"name":"future_dt_s","dtype":"f32","shape":[1]},
  {"name":"frame_index","dtype":"i64","shape":[1]}
]}
```

解析器只读 `v` / `endian` / `count` / `fields[*].{name,dtype,shape,optional}`
(`zmq_packed_message_subscriber.hpp:393-419`)。逐项说明:

* **`v`**:被解析进 `DecodedHeader::version`,但 X2 pose 解码路径**从不检查它**
  (`HandleDecoded` 里没有任何 `header.version` 引用)。写 4 或 5 都能被接受。
  本规格建议写 `5`,纯粹是为了日志可读。
* **`endian`**:必须是 `"le"`。`NeedsByteSwap()` 存在(`:170-174`),但
  `CopyFloat32IntoDouble` **无条件传 `needs_swap=false`**
  (`zmq_pose_input_source.cpp:200` 及其上方注释),`CopyInt64Scalar` 直接
  `memcpy`(`:216`)。也就是说:声明 `"be"` 不会被拒,而是被静默当成 `"le"` 解读 →
  全字段数值垃圾。**必须发小端字节,且声明 `"le"`。**
* **`count`**:被解析但 pose 路径不使用。写 `1`。
* **`fields[*].optional`**:被解析进 `FieldInfo::optional`,但**没有任何代码读它**。
  它不能用来省略字节——见下条。
* **`shape` 必须显式给出,标量也要写 `[1]`**。`ComputeByteSize()` 在
  `shape` 为空时返回 **0**(`:154-156`),于是该字段在 offset 走位里占 0 字节,
  后续所有字段全部错位 → 静默垃圾。同时 `CopyFloat32IntoDouble` 里空 shape 的
  元素数被算成 1(`:182-183` 的 `total = 1` 初值),会通过元素数检查但卡在
  字节数检查上(`:190`)。

### 3.1 dtype 字符串

字段名匹配用的是 dtype **字符串**精确比较,不是字节宽度:

* `joint_pos_mj` / `joint_vel_mj` / `root_quat_xyzw` / 三个 `*_future` /
  `future_dt_s` / `left_hand_joints` / `right_hand_joints` / `motion_token`
  必须是 `"f32"`(`zmq_pose_input_source.cpp:176-181`)。
  发 `"f64"` 会被拒(字段级),即使字节数自洽。
* `frame_index` 必须是 `"i64"`(`:209-213`)。

`GetElementSize()` 认识的 dtype:`f64`/`i64`=8,`f32`/`i32`=4,`i16`/`f16`=2,
`i8`/`u8`/`bool`=1,**未知 dtype 默认按 4 字节算**
(`zmq_packed_message_subscriber.hpp:145-151`)。最后这条意味着一个拼错的 dtype
不会报错,而是按 4 字节走位 → 错位。

## 4. 字段表

`必需性` 列的含义:

* **strict 必需** = `--strict-reference-velocity` 下缺失即整帧被拒
* **强烈建议** = 缺失不拒帧,但会导致语义退化(用缓存旧值 / bootstrap 值)
* **可选** = 缺失无影响
* **禁止** = 不要发

| 字段名(精确) | dtype | shape | 元素数 | 字节 | 必需性 | 含义 / 行为 |
| --- | --- | --- | --- | --- | --- | --- |
| `joint_pos_mj` | `f32` | `[31]` | 31 | 124 | **strict 必需** | 当前帧 body 参考位姿,弧度,MuJoCo/MJCF 关节序。**这是唯一的 "body 帧" 触发器**:缺它则整个 strict 校验分支不执行,消息被当作 token-only 帧,`body_frames_received()` 不推进、`LastReceivedMonotonicS()` 不刷新(`cpp:269-272, 338, 440-455`)。 |
| `joint_vel_mj` | `f32` | `[31]` | 31 | 124 | **strict 必需** | v5.1 新增。发送端显式计算;`fixed` 为 `(pos[t]-pos[t-1])*50`,`timestamp` 为 `delta/dt`。C++ 原样转发进 `latest_window_[0].joint_vel_mj`,不重估(`cpp:319-324, 373-375`)。 |
| `root_quat_xyzw` | `f32` | `[4]` | 4 | 16 | 强烈建议 | 根姿态,**xyzw 顺序**(scipy 约定,`reference_motion.hpp:42`)。strict 校验**没有**要求它;缺失时沿用缓存值(`cpp:256`),首帧则是 bootstrap 单位四元数 `{0,0,0,1}`(`cpp:118`)→ 策略会把身体朝世界 +X 拧。**发送端必须发。** |
| `joint_pos_mj_future` | `f32` | `[9,31]` | 279 | 1116 | **strict 必需** | 严格未来 9 个 slot 的位姿,slot k 对应 `t + (k+1)*future_dt_s`。行主序(C 序):`[slot0_dof0..dof30, slot1_dof0..]`(`cpp:300-306` 用 `future_jpos[0].data()` 平铺写入连续的 9×31 双精度数组)。 |
| `joint_vel_mj_future` | `f32` | `[9,31]` | 279 | 1116 | **strict 必需** | 同上布局。给了就原样用;不给则 C++ 会用 `future_dt_s` 反算差分(`cpp:425-435`)——但 strict 下不给就直接拒帧,所以那条回退路径在 strict 下不可达。 |
| `root_quat_xyzw_future` | `f32` | `[9,4]` | 36 | 144 | **strict 必需** | 同上布局,xyzw。**注意:它的合法性不参与拒帧判定**,详见 5.3。 |
| `future_dt_s` | `f32` | `[1]` | 1 | 4 | 强烈建议 | 未来 slot 间距。**只有落在 `[0.01, 1.0]` 闭区间内才被采纳**,否则静默忽略并沿用 `DT_FUTURE_REF = 0.1`(`cpp:325-331`)。必须填 `0.1`:它同时决定 `Sample(t)` 的 slot 索引计算(`cpp:497-503`),填错会让 tokenizer 取错 slot。 |
| `frame_index` | `i64` | `[1]` | 1 | 8 | 可选 | 单调发送端计数。仅在 hand 快照有效时被落进 `ZmqHandJointsSnapshot::frame_index`(`cpp:297-299, 393-396`),不参与任何校验或排序。 |
| `left_hand_joints` | `f32` | `[10]` | 10 | 40 | 可选 | **元素数必须是 10**,`DEFAULT_HAND_DOF_PER_SIDE` 是硬编码常量,不随 `--hand-dof 7` 变(`hpp:156`, `cpp:277-284`)。发 7 会字段级失败(不拒帧,手部快照不更新)。衣服通路不控手,建议**不发**。 |
| `right_hand_joints` | `f32` | `[10]` | 10 | 40 | 可选 | 同上。 |
| `motion_token` | `f32` | `[64]` | 64 | 256 | 可选(建议不发) | 被解码并缓存进 `latest_motion_token_`,但**类里没有任何读取它的接口**(`cpp:293-296, 398`;grep `latest_motion_token_` 只有写入点)。X2 deploy 从 `joint_pos_mj_future` 自己重新 tokenize,wire 上的 token 是死字段。 |
| `frame_index_future` | `i64` | `[9]` | 9 | 72 | 可选(建议不发) | `hpp:50` 声明了,但 `HandleDecoded` **根本没有这个分支**,落入"未知字段静默忽略"(`cpp:333`)。发了也只是白占 72 字节(字节数会被正确走位,不会错位)。 |
| `estop` | 任意 | 任意 | — | — | **禁止(除真 e-stop)** | **只要字段名出现就无条件闭锁 e-stop,值完全不被读取**(`cpp:265-268`)。`estop=0` 一样会闸停机器人。见 9.2。 |
| 其它任意字段 | — | — | — | — | 允许 | 未知字段名被静默忽略(`cpp:333`),但其声明的字节数**必须真实存在**,否则整包在 offset 走位阶段被丢(`subscriber.hpp:337-342`)。 |

### 4.1 关节序

`joint_pos_mj` / `joint_vel_mj` / `*_future` 的 31 个元素按
`policy_parameters.hpp:63-95` 的 `mujoco_joint_names` 排列:

```
0-5   left_hip_pitch/roll/yaw, left_knee, left_ankle_pitch/roll
6-11  right_hip_pitch/roll/yaw, right_knee, right_ankle_pitch/roll
12-14 waist_yaw, waist_pitch, waist_roll
15-21 left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_yaw/pitch/roll
22-28 right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_yaw/pitch/roll
29-30 head_yaw, head_pitch
```

### 4.2 四元数

* 顺序 **xyzw**(x, y, z, w),scipy / PKL 约定(`reference_motion.hpp:42`,
  `hpp:18`)。
* **不要求**发送端归一化:C++ 侧 `NormalizeQuatXyzw` 会就地归一化
  (`cpp:59-71`)。
* 接受的范数区间:`norm ∈ [0.5, 2.0]`,且 `norm² ` 必须有限且 `≥ 1e-6`
  (`cpp:62, 68`)。区间外 → 当前帧四元数判定失败(拒帧);未来帧四元数判定失败
  → 不拒帧但不提升窗口(见 5.3)。
* 全零四元数被明确拒绝,**不会**被替换成单位四元数(`cpp:56-58` 的注释解释了
  历史 bug)。
* HMCP 上游是 **wxyz**(`README_X2_GARMENT_LIVE.md:149`),发送端必须做
  wxyz → xyzw 重排。注意:wxyz 的单位四元数在 xyzw 下**仍然是单位范数**,所以
  范数校验**抓不到这个错**,只会表现为姿态错乱。

## 5. strict 模式拒帧 / 丢帧条件清单

分四层。只有第 3 层会计入 `rejected_frames()` 并留下 `last_reject_reason()`;
前两层是**静默**的,发送端拿不到任何反馈。

### 5.1 第 1 层:传输/分帧层静默丢弃(整包,连 `total_frames_received()` 都不加)

| # | 条件 | 来源 |
| --- | --- | --- |
| T1 | topic 前缀不匹配(`memcmp` 失败),或整包比 topic 还短 | `subscriber.hpp:273-280` |
| T2 | 剥掉 topic 后剩余长度 < 1280 | `subscriber.hpp:282-288` |
| T3 | header 前 1280 字节里的 NUL 终止字符串不是合法 JSON(`nlohmann::json::parse` 抛异常) | `subscriber.hpp:296-303, 393-419` |
| T4 | 任一字段声明的字节数超出实际载荷剩余长度(`offset + field_bytes > data_size`)→ **整包丢弃** | `subscriber.hpp:337-342` |
| T5 | `header.fields.size() != buffers.size()` | `cpp:225`(实践中不可达,`buffers` 由 `fields` 构造) |

多余的尾部字节**不会**导致丢包(只检查下界),但会浪费带宽。

### 5.2 第 2 层:字段级静默失败(该字段不生效,其余字段照常处理)

| # | 条件 | 后果 | 来源 |
| --- | --- | --- | --- |
| F1 | dtype 字符串 ≠ `"f32"`(浮点字段) | 该字段丢弃 → 若发生在 `joint_pos_mj` 上则退化成 token-only 帧;若发生在 `joint_vel_mj` 上则 strict 拒帧 R1 | `cpp:176-181` |
| F2 | dtype 字符串 ≠ `"i64"`(`frame_index`) | 该字段丢弃 | `cpp:209-213` |
| F3 | `shape` 元素数积 ≠ 期望值(31 / 4 / 279 / 36 / 1 / 10 / 64) | 该字段丢弃 | `cpp:182-189` |
| F4 | `buffer.size != 元素数 × 4`(f32) | 该字段丢弃 | `cpp:190-195` |
| F5 | `frame_index` 的 `buffer.size < 8` | 该字段丢弃 | `cpp:215` |
| F6 | `future_dt_s` 不在 `[0.01, 1.0]` | 静默沿用 0.1 | `cpp:328-330` |

F1–F5 会往 stderr 打一行 `[ZmqPoseInputSource] field '...'`,但**不会**计入
`rejected_frames()`。发送端调试时要看 deploy 的 stderr。

### 5.3 第 3 层:strict 拒帧(计入 `rejected_frames()`,写入 `last_reject_reason()`)

前置条件:`joint_pos_mj` 解码成功(`got_body == true`)。整个校验块在
`if (got_body)` 里(`cpp:338`)。

| # | `last_reject_reason()` | 条件 | 来源 |
| --- | --- | --- | --- |
| R1 | `strict mode: missing joint_vel_mj field` | `joint_vel_mj` 缺失或字段级失败 | `cpp:355-357` |
| R2 | `strict mode: incomplete future window (need joint_pos_mj_future + root_quat_xyzw_future + joint_vel_mj_future)` | 三个 `*_future` 里**任一**缺失或字段级失败 | `cpp:357-361` |
| R3 | `joint_pos_mj contains non-finite value` | 31 个位置里有 NaN / Inf | `cpp:339-341` |
| R4 | `root_quat_xyzw bad norm or non-finite` | **仅当同帧携带了 `root_quat_xyzw`**;范数不在 `[0.5, 2.0]` 或非有限 | `cpp:342-346, 59-71` |
| R5 | `joint_vel_mj contains non-finite value` | 31 个速度里有 NaN / Inf | `cpp:347-351` |

判定顺序:R3 → R4 → R5 → R1 → R2(每一步都先看 `reject_reason.empty()`),
所以 `last_reject_reason()` 指向**最靠前**的那条。

拒帧的确切后果(`cpp:363-369`):
`rejected_frames_++`、`total_frames_received_++`、写 `last_reject_reason_`、
**立即 return**。缓存、`latest_recv_`、`body_frames_received_`、
`has_future_window_` 全部不动。因此持续拒帧 = pose-ref watchdog 饿死 →
`--pose-ref-stale-s 0.5` 在 0.5 s 内触发有界返回,而**不是**静默喂错语义。

### 5.4 第 4 层:被接受但语义降级(最危险,因为无任何计数器暴露)

| # | 条件 | 后果 | 来源 |
| --- | --- | --- | --- |
| D1 | 9 个 `root_quat_xyzw_future` 里**任一**范数不在 `[0.5, 2.0]` | 帧**被接受**(`body_frames_received()` 推进、watchdog 刷新),但未来窗口**不提升**。`has_future_window_` 保持原值:首次即坏 → `Sample()` 走单帧回退;之前好过 → tokenizer 继续吃**上一帧的旧窗口**。 | `cpp:404-414, 437` |
| D2 | `joint_pos_mj_future` / `joint_vel_mj_future` / `root_quat_xyzw_future` 里含 NaN / Inf | **完全没有有限性校验**。`AllFinite` 只作用于 `joint_pos_mj` 和 `joint_vel_mj`(`cpp:339, 348`)。NaN 会直接流进 `latest_window_[1..9]` → tokenizer obs → ONNX。 | `cpp:404-436` |
| D3 | `root_quat_xyzw` 缺失 | 沿用缓存 / bootstrap 单位四元数,strict 不拒 | `cpp:256, 118` |
| D4 | `left_hand_joints` 长度是 7 而非 10 | 手部快照不更新,不拒帧 | `cpp:277-284` |

**D2 是发送端必须自己兜住的**:接收端不会帮你抓未来窗口里的 NaN。

## 6. 速度契约

发送端有两个不可隐式混用的模式。`fixed` 与 MuJoCo LiveMotion 逐字对齐;
`timestamp` 用于未重采样的实时 HMCP:

```
# 1a) MuJoCo parity
velocity_fixed[t] = (position[t] - position[t-1]) * 50.0

# 1b) 实时 HMCP
velocity_timestamp[t] = (position[t] - position[t-1]) / real_dt

# 2) live-edge clamp:未来 9 个 slot 全部重复最新帧
joint_pos_mj_future[k]  = joint_pos_mj          for k = 0..8
root_quat_xyzw_future[k]= root_quat_xyzw        for k = 0..8

# 3) 关键:速度不置零,重复最新的非零速度
joint_vel_mj_future[k]  = joint_vel_mj          for k = 0..8
```

`fixed` parity 的三点必须同时成立:

1. **乘的是常数 50,不是实测帧率**。衣服链路实际只有 33–35 Hz
   (`README_X2_GARMENT_LIVE.md:90, 213`),但 LiveMotion 的缓冲区 `fps=50` 是标称值,
   MuJoCo 那条已跑通的路径就是用 50 去乘的。改成用实测 dt 会得到**不同**的
   observation,parity 立即失效。
2. **差分发生在 clamp 之前**。先算 `(latest - previous) * 50`,再把 latest 复制到
   9 个未来 slot。顺序颠倒(先 clamp 再差分)会得到全零速度。
3. **未来 slot 的速度等于当前速度,不是 0,也不是窗口内的斜率**。
   注意由此产生的一个反直觉后果:窗口内位姿斜率是 0(9 个 slot 全同),但速度非零。
   这是**故意的**,不要"修正"它。也正因为如此,`joint_vel_mj_future` 在 strict 下
   必须显式发——C++ 的回退路径会用 `(future_jpos[k] - prev)/future_dt_s` 算出全零
   (`cpp:425-435`),那就破坏 parity 了。

C++ 侧对应行为:`got_vel == true` 时只置 `has_explicit_velocity_` 标志并原样转发
(`cpp:373-375`),wall-clock 差分分支(`cpp:376-385`)在 strict 下不可达。

实时 launcher 显式选择 `timestamp`。HMCP 头的 8 字节字段是 little-endian
float64 `send_ts`;当前衣服发送端在 UDP 发送前写 `time.time()`。首帧、时间回拨、
`dt < 5 ms` 或 `dt > 200 ms` 时速度置零并以当前位姿重建基线。接收单调时间同时
落盘,在旧 HMCP 发送端没有有效 `send_ts` 时作为降级时间源。

## 7. 发送端 checklist

发送端作者逐条自查,全过即可确保不被 strict 拒帧。

**封装**

- [ ] 一次 `socket.send(bytes)`,单 part。不用 `send_multipart`。
- [ ] 字节 = `b"pose"` + 1280 字节 header + 载荷,顺序不能变。
- [ ] header = `json.dumps(h, separators=(",",":")).encode()` 再
      `.ljust(1280, b"\x00")`。断言长度 == 1280。
- [ ] JSON 里不含非 ASCII / NUL。
- [ ] `"endian":"le"`,且所有数组用 `<f4` / `<i8` 小端 `tobytes()`。
- [ ] 载荷拼接顺序 == header `fields` 数组顺序,逐字段字节数与
      `shape × dtype 宽度` 完全一致,无 padding。
- [ ] 每个字段都显式写 `shape`,标量写 `[1]`,绝不省略。
- [ ] PUB `bind` 之后 `sleep ≥ 0.2 s` 再发第一帧(slow-joiner,
      `x2_zmq_protocol.md:206-218`)。

**字段**

- [ ] 字段名逐字符核对:`joint_pos_mj` / `joint_vel_mj` / `root_quat_xyzw` /
      `joint_pos_mj_future` / `joint_vel_mj_future` / `root_quat_xyzw_future` /
      `future_dt_s`。全小写下划线,`mj` 在 `future` 之前。
- [ ] 五个 strict 必需字段一个不少:`joint_pos_mj`、`joint_vel_mj`、
      `joint_pos_mj_future`、`joint_vel_mj_future`、`root_quat_xyzw_future`。
- [ ] `root_quat_xyzw` 也要发(strict 不查,但缺了姿态就是错的)。
- [ ] dtype 全部 `"f32"`;只有 `frame_index` 是 `"i64"`。**不要**用 f64。
- [ ] 未来数组 shape 是 `[9, 31]` / `[9, 4]`,C 行主序(`np.ascontiguousarray`)。
- [ ] `future_dt_s = 0.1`(必须落在 `[0.01, 1.0]`,否则被静默改回 0.1)。
- [ ] **载荷里绝对不能出现名为 `estop` 的字段**,除非真的要闸停机器人。
      检查任何"把整个 dict 打包"的代码路径不会顺手带上它。
- [ ] 不发 `motion_token`(死字段)、`frame_index_future`(未解析)。
- [ ] 若发手部:必须 10 个元素,不是 7。衣服通路建议不发。

**数值**

- [ ] 31 维按 `mujoco_joint_names` 顺序,单位弧度。
- [ ] 四元数是 **xyzw**。若上游是 HMCP 的 wxyz,做过重排。
      (范数校验抓不到 wxyz/xyzw 混淆,只能靠代码审查。)
- [ ] 四元数范数落在 `[0.5, 2.0]`,建议发送前就归一化到 1。
      9 个未来四元数**每一个**都要合格(否则窗口静默不提升)。
- [ ] `np.isfinite(...).all()` 覆盖**全部** 6 个浮点数组,尤其是三个
      `*_future`——接收端不查它们(D2)。
- [ ] 明确声明速度模式:回放/parity 用固定 50;实时 HMCP 用协议时间戳 `dt`。
- [ ] 差分在 clamp 之前算;9 个未来 slot 重复最新位姿 + **最新的非零速度**。
- [ ] 首帧没有 `pos[t-1]`:速度填 0,并且这一帧照常发(strict 只要求字段存在,
      不要求非零)。

**断流语义**

- [ ] 衣服/BLE 中断时**停止发送**,不要合成 stand 帧
      (`README_X2_GARMENT_LIVE.md:113-117`)。0.5 s 内 watchdog 会做有界返回。
- [ ] 不要靠发"退化布局"来表达"我没数据"——那会被拒帧,效果上等于断流,
      但会污染 `rejected_frames()` 计数,掩盖真实的格式 bug。

**上线前**

- [ ] 用 `scripts/garment_zmq/verify_v51_bytes.py` 跑一遍自己产出的字节。
- [ ] deploy 起来后核对三个计数器:`body_frames_received()` 在涨、
      `rejected_frames()` 为 0、`has_explicit_reference_velocity()` 为 true。
      任一不满足先看 deploy stderr 里的 `[ZmqPoseInputSource] field '...'` 行。

## 8. 参考实现的运行方式

```bash
# 合成正弦动作,50 Hz,10 s(无衣服时的通路自检)
python scripts/garment_zmq/publish_v51_reference.py \
    --host 0.0.0.0 --port 5556 --topic pose --rate 50 --duration 10

# 故意不发速度:验证 strict 拒帧(rejected_frames() 应该涨,body 帧不涨)
python scripts/garment_zmq/publish_v51_reference.py --drop-velocity --duration 5

# 故意中途停:验证 0.5 s watchdog 有界返回
python scripts/garment_zmq/publish_v51_reference.py --duration 30 --stop-after-s 8

# 纯字节自校验,不需要 pyzmq
python scripts/garment_zmq/verify_v51_bytes.py
```

## 9. 发现的风险点与未确认项

按严重程度排序。**这一节比前面所有内容都重要。**

### 9.1 `* 50` 与实际 28–35 Hz 输入速率的关系:**已拆分模式,待 live A/B**

MuJoCo LiveMotion 已确认按每个到达帧占一个标称 50 Hz 槽位追加,所以 `fixed`
必须保留用于 parity。但未重采样的真衣服帧若直接乘 50,会放大物理速度:

* **(a) 每收到一个新衣服帧就发一帧 ZMQ**(≈35 Hz):相邻 ZMQ 帧之间位姿差是
  1/35 s 的真实位移,乘 50 得到的速度被放大 ≈1.43 倍。
* **(b) 固定 50 Hz 发 ZMQ**,衣服没更新就重发上一帧:则约 30% 的帧位姿差为 0,
  速度呈 0 / 非零交替抖动。

当前实现不再二选一猜测:`fixed` 保持该仿真行为,实时 launcher 选择
`timestamp`。下一次相同动作的 dry-run 必须比较 action clip ticks、最大 raw
action 和 derivative reset 数;在这组证据出来前不能把语义修正当作真机就绪。

### 9.2 `estop` 字段:**接收端只看字段名,不看值** — 高危

`cpp:265-268` 无条件 `estop_requested_.store(true)`,而且是**永久闭锁**。
任何"把 payload dict 整体打包"的发送端,只要 dict 里有一个 `estop: 0`
或 `estop: False`,机器人就会立刻进 stage-2 纯阻尼。

`pack_pose_message`(`zmq_planner_sender.py:188`)正是"遍历 dict 全部打包"的实现。
如果衣服发送端复用它并且上游 snapshot 里带 estop 键,风险就实现了。

**建议**:发送端用显式白名单构造字段列表,不要遍历上游 dict。
参考 publisher 这样做了。

### 9.3 未来窗口无有限性校验 — 中危

D2:`joint_pos_mj_future` / `joint_vel_mj_future` / `root_quat_xyzw_future` 的
NaN 会**被接受并流进 ONNX**。接收端的 `AllFinite` 只覆盖当前帧两个数组。
发送端必须自己 `np.isfinite` 全查。(这是接收端的一个真实缺口,但按硬约束
本轮不改 C++。)

### 9.4 未来四元数坏值 → 静默用旧窗口 — 中危

D1:未来四元数范数不合格时,帧被接受、watchdog 被刷新、
`body_frames_received()` 推进,但窗口不提升 → tokenizer 吃**上一帧的旧未来窗口**。
所有计数器看起来都健康,没有任何 reject reason。这是最难排查的失效模式。

### 9.5 `root_quat_xyzw` 在 strict 下并非必需 — 中危

strict 校验块只查 `joint_vel_mj` 和三个 `*_future`(`cpp:354-361`),
**没查 `root_quat_xyzw`**。一个只发未来四元数、忘发当前四元数的发送端会被接受,
slot 0 的姿态是 bootstrap 单位四元数(`cpp:118`),而 slot 1..9 是真实姿态 →
窗口第一个 slot 姿态跳变。发送端必须自觉发。

### 9.6 wxyz / xyzw 混淆无法被自动检测 — 中危

`NormalizeQuatXyzw` 只查范数。wxyz 顺序的单位四元数范数同样是 1,
所以顺序错了**一定能通过校验**,只表现为姿态错乱。
`cpp:64-67` 的注释明确承认了这一点。HMCP 上游是 wxyz,必须重排。
**唯一防线是代码审查 + 一次静态姿态目视比对。**

### 9.7 协议总览文档与 X2 解码器不一致 — 低危(文档层面)

* `x2_zmq_protocol.md:139-140` 说 hand joints 可以是 `(7,)` 或 `(10,)`;
  X2 解码器硬编码 10(`hpp:156`),7 会被静默丢弃。
* `hpp:50` 声明 `frame_index_future`;`HandleDecoded` 里没有该分支。
* `x2_zmq_protocol.md` 完全没有 v5 / v5.1 字段。
* `x2_sonic_runtime_architecture.md:52` 提到端口 5558(PC2 那套栈),
  本通路是 5556。别照抄。

**需要谁确认**:协议文档 owner 决定是否回补 v5.1 章节。
本文在此之前是 X2 strict 通路的唯一权威。

### 9.8 `v` 字段不被校验 — 低危

X2 pose 路径从不检查 `header.version`,所以协议版本号没有任何保护作用。
发送端写错版本号不会被拒。**不要**依赖版本号做兼容性分流。

### 9.9 `conflate=1` 下的隐含约束 — 低危

`cpp:89-92` 设了 `conflate=1` + `rcv_hwm=1`。发送端如果一次 burst 多帧,
只有最后一帧会被解码,中间帧对 `body_frames_received()` 无贡献。
`--zmq-warmup-body-frames 40` 需要 40 个**被真正解码**的 body 帧,
所以 burst 补帧不能加速预热。稳定的 50 Hz(或 35 Hz)节奏比 burst 更重要。

### 9.10 未确认:衣服链路 33–35 Hz 能否满足入口闸门

`--zmq-entry-max-age-s 0.5` / `--zmq-entry-min-fresh-s 0.8` /
`--pose-ref-stale-s 0.5` 在 35 Hz(28.6 ms 间隔)下有充足余量,
但 BLE 抖动的实际尾部延迟未测。**需要衣服线给出 p99 帧间隔**。
