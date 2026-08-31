# X2 Sonic tokenizer 参考语义对齐审计(MuJoCo 已跑通路径 vs 机器人侧 C++ 部署)

生成: 2026-08-29 · 方式: **纯静态代码对比**,未运行 MuJoCo、未连接机器人、未触碰硬件。

## 0. 阅读说明

### 0.1 路径缩写(下文所有 `file:line` 均基于此)

| 缩写 | 绝对路径 |
| --- | --- |
| `PY_LIVE` | `/Users/yu/projects/humanplus_x2_bridge/x2_realtime_closed_loop.py` |
| `PY_EVAL` | `/Users/yu/projects/humanplus_x2_bridge/eval_official_sonic_x2.py` |
| `PY_GARMENT` | `/Users/yu/Documents/ChatGPT/X2/.remote_work/garment_udp_v2_reconnect_safe.py` |
| `SH` | `/Users/yu/Documents/ChatGPT/X2/.remote_work/run_v2_sim_shadow.sh` |
| `MJ_EVAL` | `/Users/yu/projects/sonic_x2_transfer_v2/scripts/eval_x2_mujoco.py` |
| `CPP/` | `/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/` |
| `DEPLOY_ROOT/` | `/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/` |

### 0.2 两侧的实际拓扑(先确认对比对象)

MuJoCo 已跑通路径(`SH:140-185`):

```text
衣服 → GMR retarget → qpos36(G1) → HMCP UDP → PY_LIVE
   → g1_to_x2 名字映射 + pose_scale → LiveMotion 滚动缓冲
   → PY_EVAL SonicPolicy.infer(tokenizer+proprio) → ONNX → action
```

机器人侧 C++ 路径:

```text
[尚不存在的 publisher] → ZMQ v5.1 packed message → CPP/src/zmq_pose_input_source.cpp
   → ReferenceMotion::Sample(t) → CPP/src/tokenizer_obs.cpp
   → CPP/src/onnx_actor.cpp → action → target_pos_mj → 安全栈 → HAL
```

**关键结构性事实**:MuJoCo 侧把 "G1→X2 映射 + pose_scale + 速度微分 + 未来窗口
构造" 全部放在**消费端**(`PY_LIVE` / `PY_EVAL`);C++ 侧把这四件事全部推给
**发送端**(`CPP/include/zmq/zmq_pose_input_source.hpp:61-70`、`:89-92`)。因此绝
大多数 "不一致" 不是 C++ 写错了,而是**契约的另一半目前没有实现者**
(`DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:21` 明确写 "衣服 → ZMQ publisher 那一段
由另一条线并行完成,本文假定参考已可获得")。

两侧加载的是**同一个 ONNX**:`SH:89-94` 钉住 SHA256
`8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9`,
`DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:196` 记录同一串。因此凡是两侧数值不同的
常数表,**必有一侧不匹配 checkpoint 的训练配置**。

---

## 1. 逐项对比表

风险等级: **P0** = 影响上电安全 / **P1** = 会让参考语义错但不直接危险 /
**P2** = 数值有偏差、可容忍或仅影响保真度 / **OK** = 已核对一致。

| # | 项 | MuJoCo 侧做法 | C++ 侧做法 | 一致? | 风险 |
| --- | --- | --- | --- | --- | --- |
| 1 | 10 帧窗口索引与采样 | `fi = min(floor((motion_time + f*0.1)/dt), n-1)`,`dt = 1/50`(`PY_EVAL:357-358`)。`motion_time = frame/ref.fps` 且 `frame = motion.frames-1`(`PY_LIVE:197`、`:208`),所以 **10 个 slot 全部 clamp 到最新帧**,零前瞻 | `t_k = current_time + k*0.1`(`CPP/src/tokenizer_obs.cpp:106-107`),再交给源:ZMQ 源用 `k = lround((time - tick_anchor_t_)/dt)` 查 publisher 给的 10 slot 窗口(`CPP/src/zmq_pose_input_source.cpp:499-503`);PKL 源用 **loop wrap** `f_idx = (long long)(time*fps) % N`(`CPP/src/reference_motion.cpp:112-116`) | **否** | P1 |
| 2 | 参考速度语义 | 后向差分 `(jp[fi]-jp[prev_fi]) * fps`,`prev_fi = max(0, fi-1)`,`fps` 取**标称 50**(`PY_EVAL:359-363`、`PY_LIVE:42`)。clamp 后 10 个 slot **重复同一非零值**,不置零 ✅ | C++ **不自己微分**:strict 模式下缺 `joint_vel_mj` 直接拒帧(`CPP/src/zmq_pose_input_source.cpp:354-361`),显式速度原样转发(`:373-375`);非 strict 才回退 wall-clock `(pos_new-pos_cached)/dt_recv`(`:376-385`) | **契约一致,实现方缺失** | P0 |
| 3 | 四元数分量序 | 全程 WXYZ:HMCP `qpos36[3:7]`(`PY_LIVE:97`)→ `LiveMotion.append` 归一化(`PY_LIVE:51-53`)→ `quat_mul` scalar-first(`PY_EVAL:364-367`)→ `quat_to_rotmat(w,x,y,z)`(`PY_EVAL:177-187`) | `ReferenceFrame.root_quat_xyzw` 为 **XYZW**(`CPP/include/reference_motion.hpp:42`);IMU 侧 WXYZ 在 tokenizer 边界显式转换(`CPP/src/tokenizer_obs.cpp:124-128`,`CPP/include/math_utils.hpp:169-172`);入口 `NormalizeQuatXyzw` 归一化(`CPP/src/zmq_pose_input_source.cpp:59-71`、`:343`) | **数学一致,线上序有风险** | P0 |
| 4 | G1→X2 关节映射 | **按名字**查字典:`delta = {nm: g1dof[i]-G1_DEFAULT_29[i]}`,`x2[nm] = DEFAULT_ANGLES_MJ[X2_ORDER_31.index(nm)] + delta.get(nm,0.0)`(`PY_LIVE:98-100`)。waist(G1 yaw/roll/pitch `PY_LIVE:29` vs X2 yaw/pitch/roll `PY_LIVE:30`)与 wrist(G1 roll/pitch/yaw vs X2 yaw/pitch/roll)的换序**由名字查表自动吸收**;两个头关节 `delta.get()` 取不到 → 0.0 → 停在 `DEFAULT_ANGLES_MJ` 的 0.0(`PY_EVAL:117-123`) | **C++ 树内完全没有 G1→X2 映射**(grep 无 `G1_ORDER`/`g1_to_x2`/29-DOF 表)。`ReferenceFrame` 直接是 31-DOF MJ 序(`CPP/include/reference_motion.hpp:40`) | **否(缺失)** | P0 |
| 5 | pose_scale = 0.7 落点 | `x2 = DEFAULT_ANGLES_MJ + 0.7*(x2 - DEFAULT_ANGLES_MJ)`,在 delta 映射**之后**、写入缓冲**之前**(`PY_LIVE:192-193`);默认值 `PY_LIVE:111-112`,启动实参 `SH:132`、`SH:144` | **C++ 树内不存在任何参考幅度缩放**。`CPP/src/tokenizer_obs.cpp:112` 直接透传 `frame.joint_pos_mj[mj]`。包内所有 `0.7` 字面量均无关(`CPP/src/safety.cpp:210`、`:214` 是 0.7 s 计时;`CPP/include/policy_parameters.hpp:189` 是 `waist_yaw` 的 action_scale) | **否(缺失)** | P0 |
| 6 | 预热 / 缓冲 | 滚动缓冲 `max_frames=600`(`PY_LIVE:41`)、标称 `fps=50.0`(`PY_LIVE:42`)、`MIN_BUFFER_FRAMES=40`(`PY_LIVE:34`)、`STAND_STILL_TIMEOUT=0.6`(`PY_LIVE:33`);未就绪则退 `stand_motion` 且 `frame=0`(`PY_LIVE:198-204`) | `--zmq-warmup-body-frames 40`、`--zmq-entry-max-age-s 0.5`、`--zmq-entry-min-fresh-s 0.8`(`CPP/src/x2_deploy_onnx_ref.cpp:559-561`,闸门 `:2354-2392`);proprio 无最小帧数闸(首次 Append 广播填满 10 slot,`CPP/src/proprioception_buffer.cpp:30-42`);另有 2 s 软启斜坡(`CPP/src/x2_deploy_onnx_ref.cpp:193`、`CPP/src/safety.cpp:28`) | **部分一致** | P2 |
| 7 | tokenizer 是否吃 root 平移 | **不吃**。`LiveMotion` 存了 `root_pos`(`PY_LIVE:56`),但 `_tokenizer` 全函数从不引用(`PY_EVAL:351-378`) | **不吃**,而且根本没有该字段:`ReferenceFrame` 只有 `joint_pos_mj`/`joint_vel_mj`/`root_quat_xyzw`(`CPP/include/reference_motion.hpp:39-43`);X2M2 文件格式亦然(`:113-119`);proprio 无 root 项(`CPP/src/proprioception_buffer.cpp:83-87`) | **一致** | OK |
| 8 | 入口 yaw 对齐 | **两侧都没做**。`normalize_heading` 只被 `evaluate_clip` 调用(`PY_EVAL:237-252`、`:518`),live 路径不调用;`PY_GARMENT` 的 T-Pose 标定是 IMU 传感器标定(`PY_GARMENT:749-770`),不是对机器人朝向;grep `yaw`/`heading` 在 `PY_GARMENT` 中未发现对 `root_rot`/`body_quat_w` 的任何变换 | `ZmqPoseInputSource::Anchor` 是显式空实现(`CPP/include/zmq/zmq_pose_input_source.hpp:207-210`),但 deploy 仍调用它(`CPP/src/x2_deploy_onnx_ref.cpp:3197-3206`、`:2981`)并打印 `yaw_anchor_delta()`(`:3206`)→ ZMQ 路径永远打印 0.0。`PklMotionReference::Anchor` 才真做 Δyaw(`CPP/src/reference_motion.cpp:150-166`) | **一致(都不做)** | P1 |
| 9 | proprio 项序与老化 / ONNX 布局 | `_append_history` 首次 `extend([v]*10)` 广播,之后 `pop(0)+append`(`PY_EVAL:340-349`);`_proprio` 按 term-major `[ang, jp, jv, action, grav]`(`PY_EVAL:380-389`);obs = `concat([tok, prop])`(`PY_EVAL:391-410`) | `Append` 首次广播填满 10 slot(`CPP/src/proprioception_buffer.cpp:30-41`),之后覆盖最旧 + `write_idx_` 前进(`:48-53`);`GetFlat` 从 `write_idx_` 正向走 → 最旧在前(`:64-79`);term 序 `ang_vel|jpos_rel|jvel|action|grav`(`:83-87`);拼接在 `CPP/src/onnx_actor.cpp:111-113`,tokenizer 在 offset 0、proprio 在 offset 680 | **一致** | OK |
| 9b | ONNX 是否 grouped 输入 | — | **单一 flat 输入**,不是分组:`GetInputCount()==1` 强校验(`CPP/src/onnx_actor.cpp:37-46`),shape `[1,1670]`(`:67-76`、`:115`),输出 `[1,31]` IL 序(`:78-87`、`CPP/include/onnx_actor.hpp:6-7`) | **一致** | OK |
| 10 | 31 DOF 序 | `X2_ORDER_31`(`PY_LIVE:30`)== `JOINT_NAMES`(`PY_EVAL:126-139`) | `mujoco_joint_names`(`CPP/include/policy_parameters.hpp:63-95`) | **逐项一致(31/31)** | OK |
| 10b | IL↔MJ 置换表 | `IL_TO_MJ` / `MJ_TO_IL`(`PY_EVAL:82-91`) | `isaaclab_to_mujoco` / `mujoco_to_isaaclab`(`CPP/include/policy_parameters.hpp:99`、`:101`) | **逐项一致且互逆(脚本核对)** | OK |
| 11 | 680-D "buggy reshape" 布局 | `command_flat = concat([all_jp, all_jv])`,再 `out[68k:68k+62] = command_flat[62k:62k+62]`、`out[68k+62:68k+68] = ori[6k:6k+6]`(`PY_EVAL:371-377`) | `is_jpos_row = (k<5)`,`pair_lo = k<5 ? 2k : 2(k-5)`,`pair_hi = pair_lo+1`(`CPP/src/tokenizer_obs.cpp:136-145`) | **一致(逐字节手推核对)** | OK |
| 12 | 6D 旋转表示 | `[r00, r01, r10, r11, r20, r21]`(`PY_EVAL:370`) | `{m[0], m[1], m[3], m[4], m[6], m[7]}`(`CPP/include/math_utils.hpp:228-233`,行主序 `out[row*3+col]` 见 `:181`) | **一致** | OK |
| 13 | 相对旋转 / 重力投影数学 | `rel = conj(base_q) * ref_q`(`PY_EVAL:364-367`);`quat_rotate_inv` = `v - w*t + cross(u,t)`(`PY_EVAL:159-174`) | `rel_xyzw = quat_mul_xyzw(cur_quat_inv, ref_quat)`(`CPP/src/tokenizer_obs.cpp:124-128`);`quat_rotate_inverse_wxyz` 同式(`CPP/include/math_utils.hpp:28-46`);重力 `quat_rotate_inverse(q,{0,0,-1})`(`:141-145`) | **代数等价** | OK |
| 14 | action_scale 表 | `ACTION_SCALE_MJ`(`PY_EVAL:110-116`),waist pitch/roll = `0.631551`,wrist pitch/roll = `0.089401` | `x2_action_scale`(`CPP/include/policy_parameters.hpp:176-208`),waist pitch/roll = `0.8420684427`(`:190-191`),wrist pitch/roll = `0.07152083551`(`:197-198`、`:204-205`) | **否(6 项)** | P0 |
| 15 | kp / kd / default_angles | `KP_MJ`/`KD_MJ`(`PY_EVAL:96-109`)、`DEFAULT_ANGLES_MJ`(`PY_EVAL:117-123`) | `kps`/`kds`(`CPP/include/policy_parameters.hpp:105-172`)、`default_angles`(`:212-244`) | **逐项一致(0 项偏差,脚本核对)** | OK |
| 16 | action clip 与 last_action 存什么 | `a = clip(action[MJ_TO_IL[mj]], ±20)`,`last_action_mj[mj] = a`,`targets[mj] = a*scale+default`(`PY_EVAL:412-420`,`ACTION_CLIP=20.0` 见 `:51-59`) | clip ±20 就地作用于 `action_il`(`CPP/src/x2_deploy_onnx_ref.cpp:353`、`:3739-3751`);`target_pos_mj[mj] = default_angles[mj] + action_il[mujoco_to_isaaclab[mj]] * x2_action_scale[mj]`(`:3776-3781`);`last_action_il_ = action_il`(即 clip 后原始动作)(`:3959-3963`) | **一致** | OK |
| 17 | 冻结腕部的实现 | `action[X2_ORDER_31.index(nm)] = 0.0`,对 6 个腕名(`PY_LIVE:209-213`),默认开启(`PY_LIVE:113-114`)。但 `action` 是 **IL 序**(由 `PY_EVAL:415` 的 `action[MJ_TO_IL[mj]]` 证明),这里用的是 **MJ 下标** | `ApplyWristFreeze(target_pos_mj)`:对 MJ `{19,20,21,26,27,28}` 置 `default_angles[mj]`(`CPP/include/wrist_bypass.hpp:56`、`:81-90`),作用在 **target**、用 **MJ 下标**,且在 action→target 之后(`CPP/src/x2_deploy_onnx_ref.cpp:3790-3793`) | **否(Python 侧索引空间错)** | P0 |
| 18 | 参考时钟来源 | `motion_time = frame / ref.fps` = `(frames-1)/50`,由**收到的帧数**驱动(`PY_LIVE:197`、`:208`) | `policy_time = now - control_entry_s_`,`now = steady_clock`(`CPP/src/x2_deploy_onnx_ref.cpp:2421-2426`、`:3693`);`control_tick_` 存在但只用于日志(`:3996`、`:4500`) | **否** | P1 |
| 19 | base 姿态来源 | sim 后端读 `data.qpos[3:7]`/`data.qvel`(`PY_EVAL:391-410`);aimrt 后端只写 `data.qpos[7:38]` 后 `mj_forward`(`PY_LIVE:224`),因此 `qpos[3:7]` 恒为 `[1,0,0,0]`、`qvel` 恒为 0(`PY_LIVE:128-135`) | 默认用 **躯干 IMU 经 3 个腰关节反推的骨盆状态**(`CPP/src/aimdk_io.cpp:190-203`,`CPP/include/math_utils.hpp:106-137`),`--raw-torso-imu` 才用原始躯干(`CPP/src/x2_deploy_onnx_ref.cpp:427`、`:1624`) | **否** | P1 |
| 20 | 头关节处理 | 恒为训练默认 0.0(见第 4 项) | 默认不动(`--head-bypass=off`);开 `Ref` 则用 `ref.joint_pos_mj[29..30]` 覆盖 target(`CPP/include/head_bypass.hpp:17-29`,`CPP/src/x2_deploy_onnx_ref.cpp:3807-3814`) | **默认一致** | OK |

---

## 2. 确认不一致的地方(按严重程度排序)

### P0-1 · action_scale 表两侧不同,腰 pitch/roll 差 1.3333×

**数值**(脚本逐项核对,`default_angles`/`kps`/`kds` 均 0 项偏差,只有 `action_scale` 6 项):

| MJ idx | 关节 | C++ (`CPP/include/policy_parameters.hpp`) | Python (`PY_EVAL:110-116`) | C++/Py |
| --- | --- | --- | --- | --- |
| 13 | `waist_pitch_joint` | `0.8420684427` (`:190`) | `0.631551` | **1.3333** |
| 14 | `waist_roll_joint` | `0.8420684427` (`:191`) | `0.631551` | **1.3333** |
| 20 | `left_wrist_pitch_joint` | `0.07152083551` (`:197`) | `0.089401` | 0.8000 |
| 21 | `left_wrist_roll_joint` | `0.07152083551` (`:198`) | `0.089401` | 0.8000 |
| 27 | `right_wrist_pitch_joint` | `0.07152083551` (`:204`) | `0.089401` | 0.8000 |
| 28 | `right_wrist_roll_joint` | `0.07152083551` (`:205`) | `0.089401` | 0.8000 |

**根因已定位**,不是随机漂移。`action_scale[i] = 0.25 * effort_limit[i] / kp_train[i]`
(`MJ_EVAL:201-203`)。反解 effort:C++ 隐含 waist 48 / wrist 4.8,Python 隐含
waist 36 / wrist 6.0。而 `MJ_EVAL:90-93` 的注释写得很明确:

> Values below = x2_ultra.py after the 2026-07-14 motor-datasheet fix
> (waist pitch/roll 36 physical peak, wrist pitch/roll 6 per PFP-41-50).
> Checkpoints trained BEFORE that fix need waist 48 / wrist 4.8 here.

即 C++ 头文件是 **2026-07-14 修正前**的表,Python 是**修正后**的表。
`CPP/include/policy_parameters.hpp:6` 标注 "AUTOGENERATED — DO NOT EDIT BY HAND",
`:11` 标注 source of truth 是 `gear_sonic/scripts/eval_x2_mujoco.py` —— 说明这份头
文件生成于该修正之前,之后没有重新生成。

**数值后果**:同一个 action 值,C++ 给出的腰 pitch/roll 目标偏离量比 MuJoCo 大
33.3%。以 `--max-target-dev` waist 上限 0.20 rad(`DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:137`)
为参考,在 action 达到 0.24 左右时 C++ 就撞上 waist 偏差上限,而 MuJoCo 侧要到
0.32 才撞上;换算成角度,MuJoCo 侧 5° 的腰部动作在机器人上会变成 6.7°。

**是否影响上电安全: 是。** 腰 pitch/roll 是躯干平衡链上的关节,幅度整体放大 1/3
会直接改变质心轨迹和踝部的补偿需求,而这一路径**从未上电验证过**
(`DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:9-12`)。两侧共用同一 ONNX(§0.2),所以
必有一侧与 checkpoint 训练配置不符,**上电前必须先确定 `x2_sonic_frozen_g1core_lora_v2.onnx`
是在 2026-07-14 之前还是之后训练的**。腕 pitch/roll 的 0.8× 方向相反(C++ 更保守),
危险性低,但同样说明表的来源版本不对。

---

### P0-2 · Python 侧 `--freeze-wrist` 索引空间错误 —— "已跑通" 的 MuJoCo 语义并不是 C++ 实现的语义

`PY_LIVE:209-213` 对 6 个腕关节名做 `action[X2_ORDER_31.index(nm)] = 0.0`。
但 `action` 是 **ONNX 原始输出 = IL 序**,这一点由 `PY_EVAL:415` 的
`action[MJ_TO_IL[mj]]` 唯一确定。`X2_ORDER_31.index(nm)` 给出的是 **MJ 下标**
`{19,20,21,26,27,28}`,被当成 IL 下标使用。

**脚本核对结果**——被置零的 IL slot 实际对应的 MJ 关节:

| 用作 IL 的下标 | 实际 MJ idx | 实际关节 |
| --- | --- | --- |
| 19 | 5 | `left_ankle_roll_joint` |
| 20 | 11 | `right_ankle_roll_joint` |
| 21 | 17 | `left_shoulder_yaw_joint` |
| 26 | 26 | `right_wrist_yaw_joint` |
| 27 | 20 | `left_wrist_pitch_joint` |
| 28 | 27 | `right_wrist_pitch_joint` |

正确的 6 个腕关节 IL 下标应为 `{25,26,27,28,29,30}`。

**数值后果**:MuJoCo 那次 "跑通" 实际上是
(a) **把两个踝 roll 和左肩 yaw 的 action 强制置零**——踝 roll 是横向平衡的主要执行器;
(b) 漏掉了 `left_wrist_yaw(19)`、`left_wrist_roll(21)`、`right_wrist_roll(28)` 三个腕关节,它们仍然自由;
(c) 因为置零发生在 `action_to_targets` **之前**,这些零会经 `PY_EVAL:415-416` 写进
`last_action_mj`,进而污染 proprioception 里 310-D 的 last_action 块——**观测也被改了**,不只是输出。

C++ 侧 `ApplyWristFreeze`(`CPP/include/wrist_bypass.hpp:81-90`)作用于
`target_pos_mj`、用正确的 MJ 下标、且在 `last_action_il_` 存储之前但不影响它
(`CPP/src/x2_deploy_onnx_ref.cpp:3790-3793` vs `:3959-3963` 存的是 `action_il`)。
所以 C++ 既不误伤平衡关节、也不污染动作历史。

**是否影响上电安全: 是,但方向特殊。** C++ 的实现是对的;危险在于**不能把
MuJoCo 的稳定性证据迁移到机器人上**。MuJoCo 里踝 roll 一直被夹死,策略实际是在
一个自由度更少的系统上表现良好;机器人上踝 roll 会全权交给策略。上电前必须**在
MuJoCo 侧先修正这个索引(或改成和 C++ 一样作用于 target)重跑一次**,否则第一次
上电就是在验证一个从未在仿真里跑过的控制配置。

---

### P0-3 · C++ v5.1 契约目前**没有任何实现者**;唯一存在的 garment publisher 字段完全不兼容

C++ strict 模式(`--strict-reference-velocity`,`CPP/src/x2_deploy_onnx_ref.cpp:558`、
`:1159`)要求每帧必须带齐:`joint_vel_mj` + `joint_pos_mj_future` +
`root_quat_xyzw_future` + `joint_vel_mj_future`,否则整帧丢弃
(`CPP/src/zmq_pose_input_source.cpp:354-361`)。解码器认的字段名见
`CPP/src/zmq_pose_input_source.cpp:269`、`:273`、`:300`、`:307`、`:312`、`:319`、`:325`。

现存唯一的 garment ZMQ publisher `_Gr00TZmqPosePublisher.publish_motion`
(`PY_GARMENT:481-522`)发的是:

| C++ 期望 | publisher 实际发 | 结论 |
| --- | --- | --- |
| `joint_pos_mj` float32[31],X2 MJ 序 | `joint_pos` (1,29),**G1 IL 序**(`PY_GARMENT:494`、`:505`) | 名字错 + DOF 数错 + 序错 |
| `root_quat_xyzw` float32[4] XYZW | `body_quat_w` (1,4) **WXYZ**(`PY_GARMENT:492`、`:504`) | 名字错 + 分量序错 |
| `joint_vel_mj` = `(pos[t]-pos[t-1])*50` | `joint_vel` (1,29),**wall-clock dt + EMA 平滑**(`PY_GARMENT:463-479`、`:506`) | 名字错 + 语义错 |
| `joint_pos_mj_future` / `root_quat_xyzw_future` / `joint_vel_mj_future` / `future_dt_s` | 均**不发** | 缺失 |
| `frame_index` | `frame_index`(`PY_GARMENT:507`) | 唯一对上的字段 |

而 `SH:168-185` 实际启动 garment 时用的是 `--stream-mode udp`,走
`_HgptUdpPublisher`(`PY_GARMENT:525-561`),发的是裸 `qpos_mj[:36]`
(`encode_frame`),速度位置返回 `np.zeros(29)` 占位(`PY_GARMENT:561`)。
**也就是说 MuJoCo 那条跑通的链路根本没用过 ZMQ。**

**数值后果**:strict 模式下**每一帧都会被拒**,`body_frames_received()` 永不增长
(`CPP/src/zmq_pose_input_source.cpp:363-369`),第 2 项入口闸(40 帧)永远不过
(`DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:69`),`policy` 请求被拒。这是 **fail-closed**,
本身安全。

**是否影响上电安全: 设计上已 fail-closed,但有一条静默失效路径。** 若有人为了
"先让它动起来" 关掉 `--strict-reference-velocity`,C++ 会走 wall-clock 回退
(`CPP/src/zmq_pose_input_source.cpp:376-385`),此时:
(a) 速度语义变成网络抖动的函数;
(b) `NormalizeQuatXyzw`(`CPP/src/zmq_pose_input_source.cpp:59-71`)**抓不住
WXYZ/XYZW 互换**——它自己的注释 `:65-67` 就承认 "a WXYZ published where XYZW is
expected would still be unit-norm";
(c) 一个把 WXYZ 当 XYZW 读的根四元数,会让 `rel = inv(cur)*ref` 变成一个**大角度
常量姿态误差**,策略会持续输出躯干扭转/前倾力矩去追一个不存在的目标姿态。
**这是本次审计里最可能造成实际摔机的单点。**

---

### P1-1 · 未来窗口的"设计意图"两份文档互相矛盾

- `CPP/include/zmq/zmq_pose_input_source.hpp:30-40` 把 "policy 看到同一帧复制 10 遍"
  定性为 v4 **bug**,说它 "destroying the look-ahead cue",并给出了历史 root-cause
  (侧步在原地抖)。
- `DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:33-39` 反过来要求 publisher "clamp 到最新帧后,
  未来 9 个 slot **重复最新位姿**"。
- 而实测已跑通的 MuJoCo 侧稳态就是**前者所批评的那个行为**:`PY_LIVE:197` +
  `PY_EVAL:357-358` 使 10 个 slot 全部 clamp 到最新帧(n=40/100/599/600 均已数值验证)。

**数值后果**:如果 publisher 按 hpp 的意图去合成真实前瞻窗口,tokenizer 的
620-D command 块会与 MuJoCo 完全不同(MuJoCo 是 10 行同值,前瞻版是 10 行递变),
两侧首帧 action 不可能对上。如果按 README 的意图,则与 MuJoCo 一致但放弃前瞻。
**必须先裁定用哪一个,再实现 publisher。** 从 "复现已跑通语义" 的目标看应选 README。

---

### P1-2 · 参考时钟:MuJoCo 由"帧数"驱动,C++ 由 steady_clock 驱动

MuJoCo: `motion_time = frame/ref.fps = (motion.frames-1)/50`(`PY_LIVE:197`、`:208`)
—— **只由收到多少帧决定**,与真实时间无关。丢包只会让参考停住,不会跳。

C++: `policy_time = now - control_entry_s_`,`now = steady_clock`
(`CPP/src/x2_deploy_onnx_ref.cpp:2421-2426`、`:3693`)。`control_tick_` 存在
(`:4336`、`:3996`)但只进日志和 obs dump(`:4500`),不参与参考时钟。

**数值后果**:定时器抖动、与 250/500 Hz writer 的执行器竞争、以及 `:3646-3650` 的
`!fresh` 早退(该 tick 完全跳过,但 wall clock 继续走)都会直接移动参考窗口的相位。
MuJoCo 侧无法复现这种相位漂移。在 clamp-to-latest 的语义下(§P1-1 选 README)这个
差异影响很小(窗口内 10 slot 同值,相位无所谓);一旦启用真前瞻窗口,它就变成
每 tick 若干 ms 的参考抖动。

---

### P1-3 · base 姿态来源不同:MuJoCo aimrt 模式喂的是理想值

`PY_LIVE:224`(aimrt 后端)只写 `data.qpos[7:38] = targets` 然后 `mj_forward`,
`data.qpos[3:7]` 保持 `spawn_stand()` 给的 `[1,0,0,0]`、`data.qvel` 保持全零
(`PY_LIVE:128-135`)。因此该模式下 `PY_EVAL:391-410` 读到的是:
**恒等姿态、零角速度、目标镜像的关节位置、零关节速度**。

C++ 默认读的是**躯干 IMU 经 3 个腰关节反推出的骨盆姿态与角速度**
(`CPP/src/aimdk_io.cpp:190-203`,反推实现 `CPP/include/math_utils.hpp:106-137`;
`--raw-torso-imu` 关闭该反推,`CPP/src/x2_deploy_onnx_ref.cpp:427`、`:1624`)。
Python 侧**没有任何对应的骨盆反推实现**(grep 无对应函数)。

**数值后果**:`--backend sim` 模式两侧可比(都用真姿态);`--backend aimrt` 模式的
观测是理想化的,proprio 里 ang_vel(30)、gravity(30)、jvel(310) 三个块共 370 维与
真机完全不同,不能作为对齐基线。**做数值对比必须用 `--backend sim`。**

---

### P1-4 · `Anchor()` 在 ZMQ 路径上是空实现,但日志会让人误以为对齐过了

`ZmqPoseInputSource::Anchor` 空实现(`CPP/include/zmq/zmq_pose_input_source.hpp:207-210`),
deploy 却在进 CONTROL 时调用(`CPP/src/x2_deploy_onnx_ref.cpp:3197-3206`)、进
SUPPORTED_POLICY 时调用(`:2981`),并打印 `yaw_anchor_delta() * 180/PI`(`:3206`)
—— ZMQ 路径下**恒为 0.0**。另外 SAFE_IDLE→CONTROL 重入路径(`:3244-3260`)根本不
重新 Anchor。

MuJoCo 侧同样不做 yaw 对齐(见表格第 8 项),所以**两侧一致**。但这是一个**共同
的潜在风险**:衣服的 T-Pose 世界系与机器人上电时的实际朝向之间存在一个未知常量
yaw 偏差,tokenizer 的 6D ori 块(60 维)会把它当成一个恒定的姿态误差交给策略。
MuJoCo 里之所以没事,是因为 `spawn_stand()` 让机器人的 yaw 恰好是 0,与衣服采集系
碰巧接近;真机没有这个保证。

---

### P2-1 · MuJoCo 侧 clamp 存在浮点取整瑕疵,缓冲增长期会漏掉最新帧

`fi = min(int(math.floor(future_time / dt)), n-1)`(`PY_EVAL:357-358`),
`future_time = (n-1)/50.0`,`dt = 1/50.0`。理想上 `fi` 应为 `n-1`,但
`((n-1)/50.0)/(1/50.0)` 的浮点结果可能略小于 `n-1`,`floor` 后得 `n-2`。

**脚本核对**(`n` 从 40 到 600 共 561 个取值):**64 个(11.4%)** 命中 `fi = n-2`。
例:`n=48 → fi=46`(应为 47)、`n=59 → fi=57`(应为 58)、`n=511 → fi=509`(应为 510)。
最大命中值为 `n=511`;稳态 `n=600` 解析为 599,正确。命中时 **10 个 slot 全部使用
次新帧**,最新收到的那一帧被静默跳过,相当于在 35 Hz 输入下多约 29 ms 的参考滞后。
只影响缓冲增长阶段(前 600 帧 ≈ 12 s)。

**是否影响上电安全: 否。** 但做首帧数值对比时,若不复现这个瑕疵,两侧 tokenizer
会在约 11% 的样本上差一整帧,容易被误判为语义不一致。

---

### P2-2 · 开启 wrist/head bypass 时,tokenizer 窗口比 cache 晚一个控制周期

`CPP/src/x2_deploy_onnx_ref.cpp:3797` 和 `:3810` 在 tokenizer 之后又调用了
`zmq_pose_source_->Sample(policy_time)`。ZMQ 源的 tick 边界检测是
`time + 0.05 < prev_sample_t_`(`CPP/src/zmq_pose_input_source.cpp:479-481`):
这次回调把 `prev_sample_t_` 从 `policy_time+0.9` 拉回 `policy_time`,于是**下一个
tick 的 k=0 调用不再被识别为 new_tick**(`policy_time'+0.05 < policy_time` 为假),
窗口快照不刷新。

索引不会错——`tick_anchor_t_` 已被 bypass 调用重锚到本 tick 的 `policy_time`,
下一 tick 的 `k = lround(0.02/0.1) = 0`,`k=9` 处 `lround(9.2)=9`(`:499-503`)——
但 tick n 的 tokenizer 用的是 tick n-1 末尾拍下的快照,即**恒定多 20 ms 参考延迟**。
`--wrist-bypass=off` 且 `--head-bypass=off`(默认)时不触发。
另注:`:477` 的注释指向 "line 1411",与实际调用点 `:3797` 不符,注释已过期。

---

### P2-3 · 预热与新鲜度阈值的细节差异

| 项 | MuJoCo | C++ | 差异 |
| --- | --- | --- | --- |
| 最少参考帧 | 40 (`PY_LIVE:34`) | 40 (`CPP/src/x2_deploy_onnx_ref.cpp:559`) | 一致 |
| 参考超时 | 0.6 s (`PY_LIVE:33`) | 0.5 s entry / 0.5 s watchdog (`:560`、`DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:130`) | 0.1 s |
| 连续 fresh 门 | 无 | 0.8 s (`:561`) | C++ 独有 |
| 参考未就绪时的行为 | 退 `stand_motion`、`frame=0`(`PY_LIVE:198-204`) | 拒绝进策略;已在策略中则 2 s 斜坡回抓拍姿态(`DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:99-105`) | 语义不同(C++ 更保守) |
| 软启斜坡 | 无 | 2.0 s smoothstep (`:193`、`CPP/src/safety.cpp:28`) | C++ 独有 |

不影响 tokenizer 数值,只影响进入/退出时机。C++ 侧一律更保守。

---

### P2-4 · `last_action` 与实际下发指令在若干路径上脱钩(C++ 侧)

`CPP/src/x2_deploy_onnx_ref.cpp:3959-3963` 存的是 clip 后的原始 `action_il`。而
`target_pos_mj` 在存储之前还会被:软启斜坡(`CPP/src/safety.cpp:28`)、
`--max-target-dev` 夹取(`CPP/src/safety.cpp:283-285`)、输出 LPF
(`:3911-3936`)、wrist/head bypass(`:3790-3814`)改写。只有
`SUPPORTED_POLICY` 分支会从 target 反解实际动作(`:3960`,
`CPP/include/startup_ramp.hpp:151-161`)。

这与训练时 IsaacLab 的语义是**一致的**(训练时 `last_action` 也是 clip 后的
env action,注释见 `:3714-3724`),所以不是 bug;但它意味着**在软启斜坡的头 2 s
内,proprio 的 last_action 块声称的动作机器人并没有真正执行**。MuJoCo 侧无斜坡,
不存在这个窗口。做首帧对比时要么跳过前 2 s,要么设 `--ramp-seconds 0`。

---

## 3. 无法从代码判断、需要实测数字的地方

### 3.1 `x2_sonic_frozen_g1core_lora_v2.onnx` 到底匹配哪一版 action_scale(阻塞 P0-1)

代码里没有任何地方记录该 checkpoint 的训练日期或 effort-limit 配置。
`SH:89-94` 与 `DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:196` 只钉住了 SHA256。
**需要**:checkpoint 侧的训练配置快照(`x2_ultra.py` 的 `effort_limit` 那一版),
或用同一 obs 分别喂两套 scale 做静态站立扭矩对比。**这是唯一必须先解决才能上电的
外部信息。**

### 3.2 G1 与 X2 的肘关节符号约定是否真的相反

`PY_LIVE:84` 的注释写 "肘=0.6 不是 -0.6",`G1_DEFAULT_29` 里左右肘均为 `+0.6`
(`PY_LIVE:89`、`:90`,表定义起于 `:85`)。而 X2 的 `DEFAULT_ANGLES_MJ` 对应肘位为负值
(`PY_EVAL:117-123`)。由于映射走的是 delta(`PY_LIVE:98-100`),**只要 GMR 输出的
肘角符号与 `G1_DEFAULT_29` 同一约定,delta 就是对的**;但代码无法证明 GMR
(`PY_GARMENT:816` 的 `retarget.retarget()`)输出的是哪一套符号。
**需要**:录一段 T-Pose 到屈肘的输入,dump `qpos36[7+18]`(left_elbow)与
`qpos36[7+25]`(right_elbow),确认屈肘时数值往正方向走。若相反,整条手臂映射符号
全错,而 pose_scale=0.7 会掩盖一部分幅度让人误判为 "只是跟不动"。

### 3.3 定时器抖动的实际幅度(决定 P1-2 是否可忽略)

`policy_time = now - control_entry_s_`(`CPP/src/x2_deploy_onnx_ref.cpp:3693`)。
20 ms 定时器(`:2136-2138`)在与 250 Hz writer 共存时的实际 tick 间隔分布,代码里
测不出来。**需要**:机器人上空跑(不进策略)记录 `policy_time` 的一阶差分直方图。
若 p99 抖动 < 2 ms,在 clamp-to-latest 语义下可忽略。

### 3.4 骨盆反推与 MuJoCo 真值的偏差(决定 P1-3 的量级)

`x2_reconstruct_pelvis_from_torso`(`CPP/include/math_utils.hpp:106-137`)在 Python
侧无对应实现,无法静态比对。**需要**:MuJoCo 里同时取 `data.qpos[3:7]`(骨盆真值)
与由躯干 IMU + 3 个腰关节反推的结果,比 gravity 向量与 ang_vel 的差。

### 3.5 衣服采集系与机器人上电朝向之间的常量 Δyaw(P1-4)

两侧都没做这个对齐,所以差值目前是一个未知常量。**需要**:上电前用 `x2_debug`
回显读机器人 yaw,同时读 publisher 的 `root_quat` yaw,记录差值。若 > 约 10°,
tokenizer 的 60 维 ori 块会持续给出一个恒定姿态误差。

### 3.6 拒帧率与端到端延迟

`rejected_frames()`(`CPP/include/zmq/zmq_pose_input_source.hpp:241-243`)与
`last_reject_reason()`(`:261`)只能在运行时读。**需要**:mock 联调时统计拒帧率
与 `LastReceivedMonotonicS()` 的 age 分布,确认 35 Hz 输入能稳定满足 0.5 s entry
age 与 0.8 s 连续 fresh 门。

---

## 4. 建议的数值对比方案

### 4.0 前置(不做这两件事,后面的对比没有意义)

1. **裁定未来窗口语义**(§P1-1)。建议按 `DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md:33-39`:
   9 个未来 slot 重复最新位姿 + 重复最新非零速度。这是唯一能复现已跑通 MuJoCo 的选择。
2. **修正 MuJoCo 侧 `--freeze-wrist` 的索引空间**(§P0-2),否则基线本身不对。
   最小改动:把 `PY_LIVE:209-213` 的 `X2_ORDER_31.index(nm)` 换成 `MJ_TO_IL[X2_ORDER_31.index(nm)]`。
   **注意这会改变 MuJoCo 侧的行为,必须重新跑通一次再作为基线。**

### 4.1 录制同一批输入

用 `PY_GARMENT` 的 UDP 路径录一段 30 s 的 `qpos36`(G1,36 维,WXYZ 根四元数),
存成 `[N, 36]` float64 + 每帧的接收时间戳。这一段是**唯一的公共输入**:
MuJoCo 侧直接回放进 `parse_hmcp` 之后的位置;C++ 侧由一个离线 publisher 读同一
文件、做 G1→X2 映射 + pose_scale + 50 Hz 微分 + 窗口合成后按 v5.1 发出。

关键:**publisher 必须做且只做 MuJoCo 侧 `PY_LIVE:96-100` + `:192-193` +
`PY_EVAL:357-363` 这三段的等价运算**,一行不多一行不少。

### 4.2 两侧要 dump 的中间量

MuJoCo 侧(在 `PY_EVAL:378` 的 `return out` 前和 `PY_EVAL:410` 处加钩子):

| 量 | 维度 | 来源 |
| --- | --- | --- |
| `x2_ref_mj` | 31 | `PY_LIVE:193` 之后的 `x2` |
| `ref_quat_wxyz` | 4 | `PY_LIVE:191` 的 `rq`(在 `:194` 被写入缓冲) |
| `fi[0..9]` / `prev_fi[0..9]` | 10+10 | `PY_EVAL:357-358` |
| `jpos_flat` / `jvel_flat` | 310 / 310 | `PY_EVAL:361-362` |
| `ori` | 60 | `PY_EVAL:370` |
| `tok` | 680 | `PY_EVAL:378` |
| `prop` | 990 | `PY_EVAL:389` |
| `obs` | 1670 | `PY_EVAL:410` |
| `action_raw_il` | 31 | `PY_EVAL:410` 的 ONNX 输出 |
| `target_mj` | 31 | `PY_EVAL:420` |

C++ 侧:已有 obs dump 通道,字段注册在 `CPP/src/x2_deploy_onnx_ref.cpp:4499-4564`
(含 `policy_time`,`:4564`)。需要确认它同时落盘 `tok_obs`(`:3699`)、
`prop`(`:3701`)、`action_il`(clip 前后各一份,`:3706` / `:3751`)与
`target_pos_mj`(`:3781`,即 bypass 与安全栈**之前**)。另外从
`ZmqPoseInputSource` 侧 dump `tick_window_[0..9]` 的 `joint_pos_mj` /
`joint_vel_mj` / `root_quat_xyzw`(`CPP/src/zmq_pose_input_source.cpp:496-503`)。

### 4.3 比什么、容差多少

按依赖顺序逐层比,**上一层不过不要看下一层**:

| 层 | 比较对象 | 容差 | 不过的含义 |
| --- | --- | --- | --- |
| L0 | 参考关节位置:`x2_ref_mj` vs `tick_window_[0].joint_pos_mj` | `max abs ≤ 1e-6` rad | G1→X2 映射或 pose_scale 不一致(§P0-3、表 4/5 项) |
| L1 | 参考四元数:`ref_quat_wxyz` vs `wxyz_to_xyzw` 后的 `tick_window_[0].root_quat_xyzw` | `max abs ≤ 1e-6`,且**同时检查符号整体翻转** | 分量序错(§P0-3 的 c) |
| L2 | 参考速度:`jvel_flat[0:31]` vs IL 重排后的 `tick_window_[0].joint_vel_mj` | `max abs ≤ 1e-5` rad/s | 微分基准 fps 或 clamp 前后顺序不一致(表 2 项) |
| L3 | 窗口一致性:`tick_window_[1..9]` 是否与 `[0]` 逐位相等(位置与速度都要) | `bitwise 相等` | publisher 没按 clamp-repeat 语义发(§P1-1) |
| L4 | `ori` 60 维 | `max abs ≤ 1e-6` | 6D 表示或 `rel` 乘法顺序错(表 13 项) |
| L5 | `tok` 680 维 | `max abs ≤ 1e-6` | buggy-reshape 行布局错(表 11 项) |
| L6 | `prop` 990 维 | `max abs ≤ 1e-6` | 项序/老化/priming 不一致(表 9 项);**注意必须 `--backend sim`**(§P1-3),且跳过前 2 s 或设 `--ramp-seconds 0`(§P2-4) |
| L7 | `obs` 1670 维 | `max abs ≤ 1e-6` | 拼接偏移错 |
| L8 | `action_raw_il`(clip 前) | `max abs ≤ 1e-4` | 同一 ONNX 同一 obs 应几乎逐位相同;`CPP/include/onnx_actor.hpp:17-18` 记录 fused ONNX vs .pt 的历史偏差是 2.4e-7,阈值 1e-4 | 
| L9 | `target_mj`(bypass 与安全栈之前) | **先按各关节分别比**;`waist_pitch/roll(13,14)` 与 `wrist_pitch/roll(20,21,27,28)` 预计分别差 1.3333× 与 0.8×,其余 25 个关节 `max abs ≤ 1e-6` | 若偏差模式**正好是** §P0-1 那 6 项,则确认根因就是 action_scale 表版本;若还有别的关节偏差,说明另有问题 |

**float32 说明**:`PY_EVAL:352-354` 的 tokenizer 中间量是 float32,C++ 侧
`tok_obs` 也是 float32(`CPP/src/onnx_actor.cpp:111`),而 C++ 的 `ReferenceFrame`
是 double(`CPP/include/reference_motion.hpp:40-42`)。所以 L0-L2 允许到 1e-6,
L4-L7 走 float32 后 1e-6 已接近 eps 边界(1670 维中 ~1e-7 量级的差属正常),
建议同时报 `max abs` 和 `argmax` 的下标,便于定位到具体是哪一行哪一项。

### 4.4 单帧闸门(上电前的最后一道)

固定一帧输入(建议就用 T-Pose 后的第 41 帧,恰好跨过 40 帧预热门),两侧各跑一次,
要求:
1. `tok` 680 维逐位一致(≤1e-6);
2. `prop` 990 维逐位一致(≤1e-6);
3. `action_raw_il` 31 维一致(≤1e-4);
4. `target_mj` 的 25 个非争议关节一致(≤1e-6),6 个争议关节的比值**精确等于**
   1.3333.../0.8;
5. `tick_window_[1..9]` 与 `[0]` 逐位相等。

五条全过,才能认为 §0 里 "同一输入两侧参考窗口与首帧动作一致" 这个验收门通过。

---

## 5. 未找到 / 需要确认位置

- **发送端 v5.1 publisher:未找到。** 已检查 `PY_GARMENT` 全文、
  `DEPLOY_ROOT/README_X2_GARMENT_ZMQ.md`。README `:21` 自述这一段由另一条线并行完成。
  `GR00T_audit/gear_sonic/utils/teleop/online_sonic_tokenizer.py` 消费同类快照契约
  (`:262` 引用 `body_pose_q_mj` / `root_quat_xyzw` / `joint_pos_mj_future` /
  `root_quat_xyzw_future`),但它是 VLA motion-token 编码器,**不是** garment→C++ 的桥。
  该文件 `:283-335` 还记录了一条已废弃的 freeze-pose 路径(把当前姿态平铺 11 次伪造
  未来窗口)。
- **C++ 侧的 G1→X2 映射:不存在。** grep 全包未发现 29-DOF 表、`G1_ORDER`、
  `g1_to_x2` 或等价物。
- **C++ 侧的 pose_scale:不存在。** 见表 5 项。
- **Python 侧的骨盆反推:不存在。** 见 §3.4。
- **`CPP/src/x2_deploy_onnx_ref.cpp` 共 4811 行**,本审计只逐行读了 obs 组装、
  action 后处理、状态迁移与 CLI 解析相关段落(约 `:180-560`、`:1100-1700`、
  `:2100-2200`、`:2340-2400`、`:2420-2450`、`:2510-2530`、`:2670-2760`、
  `:2970-3000`、`:3160-3270`、`:3640-3830`、`:3880-4000`、`:4240-4260`、`:4490-4570`)。
  其余段落(HAL writer、日志、录制)未逐行核对。

