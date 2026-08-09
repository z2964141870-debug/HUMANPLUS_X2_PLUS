# BASE Phase23：延迟/状态门控交权 A/B

## 游戏任务

- [x] 冻结 Stage306 moving、Phase21 f005 recovery、stiff1.2、fixed upper、PD、指令、0.5 s handoff blend。
- [x] 冻结 Phase19 v2 previous-action 支持域：432 行，稳健 NN LOO-p95 `0.5483584339772765`。
- [x] 纯测试证明 gate 默认关闭；默认路径保留原交权顺序；其他 stop controller 不进入 gate。
- [x] 官方 AimDK MuJoCo fixed/gated 各 5 条，10/10 interface valid。
- [ ] stop/full 物理门通过。
- [ ] 解锁训练。

## 假设

共同坍塌主要由固定 stop+2.0 s 时，generator support phase 与 recovery actor 的 previous-action 支持域不匹配造成；延迟到“双支撑且 previous-action 回到支持域”再交权，应阻止坍塌。

## 干预

仅 candidate 启用默认关闭的 handoff gate。stop+2.0 s 后继续原 Stage306 零命令 brake；只有：

1. generator 左右 contact 均为正（注意：不是实测足底接触）；
2. pre-inference `previous_action` 到冻结 Phase19 v2 支持集的稳健最近邻距离 `<=0.548358`；

才以同一 0.5 s C2 blend 交给同一 f005 recovery actor。stop horizon 内不满足则保持 brake，绝不强制交权。

## 对照

control 在 stop+2.0 s 固定交权。两组均使用同一模型、事件、时钟、PD、动作幅值、上肢和五次 official episode；没有训练或参数扫描。

## 结果

| 指标 | fixed | gated |
|---|---:|---:|
| valid | 5/5 | 5/5 |
| stand / startup | 5/5 / 5/5 | 5/5 / 5/5 |
| move / stop / full | 0/5 / 0/5 / 0/5 | 0/5 / 0/5 / 0/5 |
| gate wait 中位数 | 0.00 s | 0.36 s（0.36–0.44） |
| handoff 中位时刻 | 2.00 s | 2.36 s |
| root-z<0.45 中位时刻 | 3.90 s | 4.72 s |
| handoff 后 root-z<0.45 中位延迟 | 1.90 s | 2.36 s |
| heading max 中位数 | 0.572 rad | 0.504 rad |
| lateral 中位数 | 0.554 m | 0.477 m |
| stop drift 中位数 | 0.274 m | 0.257 m |
| settle time 中位数 | 4.34 s | 5.14 s |

candidate 五条交权距离为 `0.360–0.547`，均小于冻结阈值，且 generator double-support 为真；slot count 与实际等待 tick 一致。门控确实生效，不是合同失效。

两组事件顺序仍然一致：先 tilt 越界，再 height collapse。门控把倒地整体推迟约 `0.82 s`，但 5/5 仍倒，不能把倒地后的较小 drift 或更晚 settle 当作恢复成功。

## 结论

这是一个“机制有连续改善、最终命题失败”的干净结果：等待支持域和 generator 双支撑能延迟坍塌，并改善部分 heading/lateral 连续量；但它没有把任何一条变成 stop/full pass。因此“previous-action 支持域 + generator DS 是安全交权的充分条件”被证伪，不能据此解锁训练。

相关性边界：当前 contact 只是 gait generator 状态，不是物理足底接触；本实验只证明冻结 gate 对时间轴有因果影响，不能把未测接触或 recovery actor 本体归为唯一根因。

## 下一步

不继续扫等待时间或阈值，也不继续 recovery 训练。下一项若继续，应只做一个更严格、仍 training-free 的可证伪门：在保留 previous-action 与 generator DS 的同时，要求 recovery actor 的完整可观测物理状态（至少 projected gravity / base angular velocity）也处于 Phase19 v2 支持域；若该条件在 stop horizon 内从不成立，应直接判定现有 Stage306 brake 无法把系统送入 recovery 的训练支持域，而不是强交权。

## 证据与边界

- 聚合 JSON：`reports/official_x2/phase23_delayed_handoff_ab.json`
- gate manifest：`manifests/x2_phase23_previous_action_support_gate.json`，文件 SHA `a097378247a6cfc1d5f25fcc97a38674576902789c13374bc9a70692c428548e`
- adapter SHA：`a02faa192b2fed9c5cac502019d724a922eaa9261bb1d699a5adafa7ac673cfa`
- 只使用官方 AimDK MuJoCo；未上真机、未训练、未改 WBT、未 Git/百度上传。
