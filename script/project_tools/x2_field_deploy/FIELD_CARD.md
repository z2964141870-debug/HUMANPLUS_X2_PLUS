# X2 键盘行走 · 现场任务书 v2（2026-10-23 实战修订）

机器人: AgiBot X2 Ultra · 策略: sonic_x2_v0.2 (HUMANPLUS_X2_PLUS)
目标: 悬空首次键盘指令行走。**全程一人手扶/急停在手, 悬空吊架下进行。**

## 终端与账号速查
| 终端 | 机器 | 账号 |
|---|---|---|
| T3 冻结 | SoC0 (10.0.1.40) | run@ |
| T1 仪式 / T2 键盘 | SoC1 (10.0.1.41) | agi@ |

注意: 机器人**每次断电重启 SSH 主机密钥都会变**, 连不上先:
`ssh-keygen -R 10.0.1.40` / `ssh-keygen -R 10.0.1.41` 然后重连输 yes。

## 0. 上电预检（推荐每次上电后跑一次, ~2 分钟, 机器人零运动）
SoC1 上:
```bash
bash ~/walk/preflight_link.sh
```
五步全 OK（kplanner 实机起 / :5556 v4 13字段帧 / 键盘实机 ingest）才继续。

## 1. T3 —— 冻结主控（SoC0, 第一个动!）
```bash
bash ~/walk/pause_mc.sh hold 7200
```
- 看到 `FROZEN pid=... state=T` = 冻结落位（默认窗口 7200s, 到点自动恢复）
- **这个终端别按 Ctrl-C**（Ctrl-C = 立即解冻）; 断 ssh 不会解冻
- 提前收工: 另开终端 `bash ~/walk/pause_mc.sh release`

## 2. T1 —— 点火仪式（SoC1）
```bash
X2_RITUAL_TUNING=/home/agi/gear-sonic/gear_sonic_deploy/configs/real_deploy_tuning/frozen_g1.yaml \
  bash ~/gear-sonic/ritual_start_kb.sh
```
预期顺序（工程师远程监视确认, 现场只看两个绿灯）:
- `pose stream OK` = 仪式过闸, 五件套(kplanner/deploy/watchdog/hand/motor)起
- 随后 ~1 分钟内 kplanner 武装（deploy 的 x2_debug 喂航向）→ deploy 进入接控
- 仪式可安全重跑: 死掉的 deploy pane 会被自动清除重启

## 3. T2 —— 键盘行走（SoC1, 新终端）
```bash
bash ~/walk/t2_keyboard.sh
```
**第一拍永远先 `空格`（全零, 站稳确认）, 再给 `w`。**

| 键 | 动作 |
|---|---|
| w / s | 前进 0.35 m/s / 后退 0.1 |
| a / d | 原地左/右转 |
| 空格 | 全零急停（意图清零） |
| q | 退出 |

## 4. 收工顺序（反着来）
1. 键盘 `q`（退出前先 `空格` 停稳）
2. deploy 停止（工程师侧确认 :5558/:5556 停流）
3. SoC0: `bash ~/walk/pause_mc.sh release` → 看到 `RELEASED (identity verified)`
4. MC 恢复出厂控制, 收工

## 5. 中止阶梯（任何异常, 从轻到重）
1. `空格`（意图清零, 每一拍都可停）
2. 键盘 `q`
3. T3 `release` 或 Ctrl-C（MC 恢复, 系统回到出厂状态）
4. **失控/碰撞: 手持真实急停**, 事后 `release` 恢复 MC

## 三条死规矩
1. **MC 冻结在前, deploy 接控在后; 收工完全相反**
2. **键盘第一拍永远是空格**（从零开始, 不给悬空机器人任何初速度）
3. **手不离急停**, 悬空全程

## 工程师侧已知项（非阻塞, 记录在案）
- deploy 已打补丁: PC2_HOST=10.0.1.41 + --no-stop-mc（原文件备份 .bak-20261023）
- x2_stand_default_pose.yaml 缺失 → HOLD_FOR_MC 用 default_angles 兜底
- 真实行走证据链 = tape 非零 intent_recv + kplanner armed 日志 + 悬空步态
