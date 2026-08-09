# X2 WBT Stance-Foot XY Lock Phase8

- 裁决：**PHASE8_STANCE_XY_LOCK_OFFLINE_REJECTED**。
- Phase7 24球、clearance、MA9、root XY/orientation、source timing 全部冻结；只新增 stance-window XY lock。
- 无 prescribed/free 物理回放、训练、teacher、checkpoint、BASE、Git、百度或真机。contact/slip 是模型估计，不是实机 GRF/COP。

## 固定结构

每个 Phase7 连续 geometry-contact 窗以首帧 ankle-roll XY 为 anchor，仅同侧6个腿关节做 DLS；窗口两端用4帧 C2 blend，摆动段关节保持逐位精确。数值常量继承 Phase4，不按 free survival 选参。IK 后只重新执行冻结的 Phase7 root-z 公式。

## 离线门

| role | stance windows L/R | joint step base→candidate/limit | model slip base→candidate/limit | contact agree | ground | terminal/swing exact | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| walk_straight | 5/7 | 0.089→0.093/0.150 | 0.355→0.515/0.390 | 0.956 | True | True/True | False |
| turn_left | 8/9 | 0.153→0.231/0.168 | 1.750→2.890/1.925 | 0.906 | True | True/True | False |
| turn_right | 9/10 | 0.147→0.243/0.162 | 1.955→3.650/2.151 | 0.810 | True | True/True | False |
| stand_to_walk | 3/5 | 0.140→0.198/0.154 | 1.234→1.901/1.357 | 0.914 | True | True/True | False |
| walk_to_stand | 4/5 | 0.140→0.215/0.154 | 1.418→2.302/1.560 | 0.885 | True | True/True | False |

## 停止裁决

- 结果：离线门在 ['walk_straight', 'turn_left', 'turn_right', 'stand_to_walk', 'walk_to_stand'] 失败；失败项为 ['contact_intent_agreement_ge_0p90', 'joint_step_within_tier_or_110pct_baseline', 'model_stance_slip_not_worse']，未进入 prescribed/free。
- 结论：碎片 stance-window 的 clipwise 常量XY anchor在C2进出边界制造更高足速和关节跳变；当前IK reference生成器表示不足。
- 下一步：冻结反例，不扩 anchor/blend/SE2/IK 参数；若继续应转为连续接触相位/全轨迹优化或闭环controller，而非逐窗IK。
- 该结果只否定当前 clipwise stance-window XY-lock 生成器表示；不否定 X2、WBT、Any2Any 或闭环控制器可实现 stance stabilization。
