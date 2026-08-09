# X2 WBT Collision Contact Schedule Phase9

- 裁决：**PHASE9_CONTACT_SCHEDULE_REJECTED**。
- Phase7 root-ground Bronze、动作、root、关节、控制和物理模型全部冻结；只处理 contact label。
- 五动作各只补录一次相同 free-root zero-update contact trace；先复现 Phase7 aggregate，之后全部离线，未做第二轮 physics。
- MuJoCo 路径无随机采样/seed；scene、control 与 Phase7 Bronze 均记录 SHA256。
- collision/contact 是官方 MuJoCo 模型估计，**不是实机 GRF、COP、足底力或 wrench 真值**。

## 预注册统一契约

- 30Hz 上 contact-on/off 均需连续 `2` 帧确认；保留状态最短 `3` 帧。
- 参数对五动作统一，不按 clip、free survival 或结果调节；schedule 不修改任何运动轨迹。
- DS/左SS/右SS/flight 仅由两脚标签确定；禁止退化成全程双接触，且必须保留原始左右单支撑语义。

## 共同存活窗结果

| role | frames | raw windows L/R | schedule windows L/R | source agree/F1 | schedule agree/F1 | SS raw→schedule→free | event max source→schedule | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walk_straight | 47 | 0/0 | 0/0 | 0.968/0.984 | 0.021/0.000 | 0.000→0.000→0.043 | 1→0 | False |
| turn_left | 59 | 2/2 | 1/0 | 0.508/0.590 | 0.263/0.068 | 0.136→0.068→0.390 | 16→17 | False |
| turn_right | 35 | 1/0 | 0/0 | 0.500/0.529 | 0.243/0.000 | 0.029→0.000→0.371 | 14→0 | False |
| stand_to_walk | 50 | 0/0 | 0/0 | 0.510/0.495 | 0.030/0.000 | 0.000→0.000→0.020 | 1→0 | False |
| walk_to_stand | 81 | 2/1 | 2/1 | 0.475/0.541 | 0.333/0.154 | 0.123→0.123→0.494 | 20→44 | False |

## 裁决

- 结果：统一 schedule 未在 ['walk_straight', 'turn_left', 'turn_right', 'stand_to_walk', 'walk_to_stand'] 相对 Phase7 source intent 一致改善；按预注册门停止。
- 结论：固定的hysteresis/min-dwell不能把prescribed collision事件统一对齐free-root realized contact；否定该temporal schedule，不修改Phase7 Bronze。
- 下一步：不扩hysteresis/dwell/clip参数；保留Phase7 Bronze，转向闭环contact-conditioned控制或具有连续相位的轨迹方法。
- 该裁决只评价统一的 temporal collision-label schedule；不评价 X2 硬件接触力，也不否定闭环控制器或其他 reference 生成方法。
