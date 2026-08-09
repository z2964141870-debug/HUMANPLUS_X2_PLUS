# BASE Phase32：closed AimDK 前缀无扰动观测

## 结论

Phase32 已打通此前缺失的观测层：在不改官方 binary、scene、controller、PD 或动作合同的情况下，`LD_PRELOAD` shim 能完整记录 closed AimDK MuJoCo 前 0.3 秒的 reset/forward/step 调用顺序和 `mjData` 物理状态。Toy 和原始官方 scene 的 hook on/off 轨迹均逐步 bitwise 相同；唯一一次 closed rollout 仍通过 Stage250 的 startup/move/stop/full 四门，因此没有检测到 observer 引入的物理扰动。

这一步没有直接锁定 Phase28 与 Stage250 分叉的唯一根因，但把 Phase31 的“closed wrapper 隐藏 prepare/integration state”从不可观测变成了可观测。下一步可以使用这次捕获的同一时刻物理状态做单次 direct snapshot/replay，而不再猜 prepare 轨迹。

## 假设

Stage250 的 closed AimDK 成功轨迹与 Phase28 direct runner 的早期分叉，可能来自可见 `q/qdot/root` 以外的 simulator integration state、reset/forward 顺序或真实 prepare 控制历史。若一个只读 observer 能在不扰动物理的前提下记录这些量，就能为后续单变量 replay 提供权威初态。

## 干预

- 新增默认关闭的 ELF interposition shim，仅拦截 `mj_resetData`、`mj_forward`、`mj_step`。
- 每次先调用原始 MuJoCo 函数，再把状态复制到预分配 mmap；超过 0.3 秒停止记录。
- 记录：调用类型/序号/线程/指针、`qpos/qvel/act/ctrl`、`qacc/qacc_warmstart`、`qfrc_actuator/qfrc_constraint`、contact geom/frame/position/wrench、`nisland/nefc`。
- 不修改 vendor 文件；shim、decoder、runner 均位于本项目 Phase32 新文件。

## 对照

1. exact vendor MuJoCo 3.3.7 toy，固定 300 步控制，hook off vs on。
2. 未修改的官方 `scene.xml`，同一重建初态和 300 步冻结控制，hook off vs on。
3. 两层无扰动门通过后，仅运行一次隔离的 closed AimDK Stage250 straight capture；ROS domain=232、无真机。

预注册门：每步 `qpos/qvel` 最大差不超过 `1e-12`；零 dropped record；observer 时间不超过 `max(2×control, control+0.1s)`。任一失败均不允许 closed capture。

## ABI 与静态审计

- Header 与 vendor runtime 均为 MuJoCo 3.3.7，`mjtNum` 编译期确认是 double。
- C/Python ABI 尺寸一致：header 256 bytes，record 19024 bytes。
- shim 不直接链接 `libmujoco`，通过 `RTLD_NEXT` 调用原始函数。
- 官方 module 对三个目标函数均保留动态 undefined symbol，具备 ELF interposition 条件。
- closed binary 不是 secure-exec/setuid 路径；容器可显式传递 `LD_PRELOAD`。

第一次 closed 启动在进入 physics 前因主机编译产物要求 `GLIBC_2.38` 被 loader 拒绝；没有产生 mmap 或 episode。随后保持 source/header 不变，在 network-disabled 官方容器（glibc 2.35）内重编译，得到只要求至 GLIBC 2.34 的兼容 binary，并重新通过 toy 与 official scene 两层无扰动门。这个 pre-physics ABI 修正已单独记录，未把历史 hash 静默覆盖。

## 结果

### 无扰动门

| 域 | qpos absmax | qvel absmax | wall ratio | trace |
|---|---:|---:|---:|---|
| toy | 0 | 0 | 1.207 | 606 committed / 0 dropped |
| official scene | 0 | 0 | 1.060 | 1252 committed / 0 dropped |

两域物理状态 bitwise 一致，性能门通过。

### closed AimDK 捕获

- Stage250 rollout：startup=true、move=true、stop=true、full=true。
- mmap：607 committed、0 dropped、单一 `mjData` pointer、sequence 连续。
- 调用计数：reset=3、forward=303、step=301。
- 最后一次 reset 位于 sequence 4；其后恰有 300 个正式物理 step，时间 0.001–0.300 秒。
- 第一条非零 `ctrl` 在 0.001 秒；第一次接触在 0.066 秒。
- adapter 前 14 行 root telemetry 与物理 step 一一匹配，统一偏移为 `physics_time = elapsed + 0.026s`，最大对齐 score 为 `4.98e-11`。

这说明此前将 adapter 的 `elapsed=0` 直接当作 MuJoCo `time=0` 是不成立的；closed wrapper 在第一条可见 telemetry 前已经执行了 26 ms 的物理与控制。

第一条完整 93D stand row（adapter index 10）对应 MuJoCo `t=0.226s`。此时隐藏状态为：

- `qacc_warmstart` L2 = 46.5006；
- `ctrl` L2 = 19.1659；
- contact=20、island=1、efc=111；
- joint q 与 93D obs 重建的 absmax 差 `2.85e-5 rad`；
- joint dq absmax 差 `3.76e-3 rad/s`；
- root XYZ 与 telemetry 完全一致。

微小 q/dq 差异符合 1 ms physics 与 ROS/float32 observation sampling 边界，不能在本阶段归因某个字段。

## 结论

1. **机制改善已成立**：closed simulator 的 reset/prepare/integration state 现在可无扰动采集。
2. **关键时间语义已纠正**：首条 telemetry 是 MuJoCo reset 后 26 ms，而首个完整 actor row是 226 ms，不是直接 runner 过去假定的“可见 row 即 t=0”。
3. **唯一根因仍未证明**：本次是新 episode，不能宣称旧 Stage250 的隐藏状态与本次 bit-identical，也不能把 `qacc_warmstart` 单独认定为主因。
4. **训练继续锁定**：Phase32 不训练，也没有为 PPO/长训提供解锁依据。

## 下一步

只做一个可证伪实验：从本次 closed 捕获中选定第一条完整 stand row，恢复 `qpos/qvel/time/ctrl/qacc_warmstart` 与相同 controller history，在 direct official MJCF 中做同状态双 fork；control 仅恢复过去使用的可见状态，candidate 恢复本次捕获的完整 integration state。比较未来 10/25/50 tick。若 candidate 显著缩小首 20 ms 分叉，才说明隐藏 integration state 是主要缺项；若仍不缩小，则转向 ROS command application timing/closed module 内部状态，不继续猜 prepare 参数。

## 证据

- 结果 JSON：`reports/official_x2/phase32_closed_prefix_observer.json`
- 预注册：`reports/official_x2/phase32_ldpreload_observer_prereg.json`
- ABI 修正 provenance：`reports/official_x2/phase32_ldpreload_observer_abi_amendment.json`
- mmap：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase32_closed_prefix/phase32_stage250_closed_prefix_once.mmap`（SHA256 `fc91073c...562e`，约 75 MiB）
- rollout：同目录 `phase32_stage250_closed_prefix_once.json`（SHA256 `45d947fd...1d0f`）
- 兼容 shim：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase32_ldpreload_observer/libx2_phase32_mjtrace_glibc235.so`（SHA256 `ec4fe929...acd3`；source/header 仍是可审计权威）
- Tests：3 passed；无训练、无 WBT、无真机、无 Git/百度操作。
