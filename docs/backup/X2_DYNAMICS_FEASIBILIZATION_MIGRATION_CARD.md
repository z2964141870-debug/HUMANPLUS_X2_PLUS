# X2 动力学可行化路线迁移卡

> 盘点日期：2026-08-10
> 目标：在新电脑/服务器复现 Stage152-B 四域基线，并继续 OmniTrack / SBTO / DDR / DSMS 式动力学可行化实验。

## 先纠正一个数据来源事实

当前四域基线使用的 `stage72_official_true_forward4_v1` 含 B、D 两段及其镜像，共 4 条动作。其 `metadata.pkl` 明确记录源为：

```text
data/processed/x2_real_readonly_session04_canonical_31dof.npz
```

因此这四条不是普通的 AMASS/SMPL → GMR 输出，而是 X2 native/readonly 31DoF 数据加工出的 reference。可以对它们做 reference physicalization、SBTO 或接触优化，但不能在论文和报告中称为“GMR 原始参考”。DDR 若要从人体关键点直接开始，需另选一条带 SMPL/人体关键点源的 matched forward-walk。

## 一、必须从 Git 恢复

```text
https://github.com/z2964141870-debug/HUMANPLUS_X2_PLUS.git
```

至少检出本卡对应提交之后的版本，并保留：

- `requirements.txt`
- `scripts/`、`src/`、`tools/`、`tests/`
- `reports/` 和 `docs/backup/manifests/`
- X2/IsaacLab 配置、冻结 gate 和 DC-PEFT 代码

Git 不包含 checkpoint、大 trace、数据集和 Docker 镜像。

## 二、百度网盘中已存在：最小必下集合

以下路径均相对于官方 `bdpan` 根 `/apps/bdpan/`；命令中不要写 `/apps/bdpan/` 前缀。

### A. 官方 X2 物理域

1. 官方 AimDK v1.0 SDK、MJCF、ROS workspace 和随包模型：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/official_x2_v1/sdk/aimdk-x2-v1.0.0-official.zip
```

远端大小：`156,945,205 B`。

2. 最新且最完整的官方 closed-loop 物理 trace、Phase50 上肢 A/B、Phase49 WBT 权重：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase55/x2_phase55_native_upper_robust_20260809.tar.gz
```

远端大小：`495,434,849 B`。其中包含 Phase34/Phase50 两份完整 mmap、对应 JSON，以及 Phase49 source/best/rejected 权重。

3. Stage219 速度型原生后端 PT/ONNX 和视频基线：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase42/x2_phase42_dual_track_20260809.tar.gz
```

远端大小：`20,567,491 B`。

### B. 当前重定向/动力学 reference 资产

4. 固定 24 条面板、Phase29 repair、Phase30 retime reference：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase30/x2_phase30_dual_track_20260809.tar.gz
```

远端大小：`18,973,586 B`。

5. Gold native motion、train/held-out MotionLib 和已有 recovery 模型：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase10/x2_phase10_recovery_native_gold_20260809.tar.gz
```

远端大小：`20,594,628 B`。

6. 若要继续 faithful WBT / exact-S7，而不只做 Stage152 四域基线，再下载：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase47/x2_phase47_official_state_faithful_s7_20260809.tar.gz
```

远端大小：`299,929,905 B`。

7. 若要复用 held-out repair、Phase32 worker shards 和 outcome-aware BASE，再下载：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase36/x2_phase36_dual_track_20260809.tar.gz
```

远端大小：`29,623,463 B`。

### C. DC-PEFT 基础权重

8. 已归档的 Stage152 双 critic / materialized 权重：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/checkpoints/stage152_B_dual_equal_split_init.pt
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/checkpoints/stage152_B_materialized_v1.pt
```

远端大小分别为 `193,683,894 B`、`151,324,629 B`。

注意：它们可用于现有 DC-PEFT 训练入口，但不是四域基线脚本默认引用的原始 `model_step_000200.pt`。

### D. 旧工程兼容快照

9. 旧 X2/SONIC 工程快照：

```text
HUMAN+/x2_teleop_final.zip
```

远端大小：`454,053,271 B`。它可恢复旧目录骨架和部分资产，但归档时间早于后续 Stage152 工作，不能替代下节列出的缺失 seed bundle。

## 三、当前百度网盘尚缺：迁移前必须补传

这是本路线真正的阻塞项。建议在当前机器打成一个新包并上传到：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/dynamics_feasibility/
```

建议包名：

```text
x2_forward4_dynamics_feasibility_seed_v1.tar.gz
```

包内必须包含：

1. 四域冻结基线的精确 source checkpoint：

```text
/home/humanplus/x2_teleop_final/x2_sonic/logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/model_step_000200.pt
```

大小 `195,741,434 B`，SHA-256：

```text
b73c345995c5d468c18d223de96b29fff4cd4866e4d56bb6d5296c4a68540679
```

同时带上同目录的 `config.yaml`、`meta.yaml`、训练诊断和加载报告；不需要重复打包 `last.pt`。

2. 四条 frozen forward-walk reference：

```text
/home/humanplus/x2_teleop_final/x2_sonic/motion_lib_x2/stage72_official_true_forward4_v1/
```

目录约 `320 KiB`，必须保留 `metadata.pkl` 与 4 个 motion PKL。

对应 SHA-256：

```text
metadata.pkl                                                        8262a6311f29dd85ecba2151180944fe8247c0ea7efe76e9136dcd19bc61f909
official_x2__19_walk_forward_short_pulses_B__10p97_15p77.pkl        ba39606bf55d08c7b89f42b1a7abcf9c83279238e86cabf2b822368f4d99cd09
official_x2__19_walk_forward_short_pulses_B__10p97_15p77__mirror.pkl 4e5fedd9004caddb28b3e48adbb393cd4532ffe3464c5ded483fc74de0552858
official_x2__19_walk_forward_short_pulses_D__15p00_19p80.pkl        17ccc5129b80517675d8e311259650c0cce238e4a533673d25d46ff4f387a8c9
official_x2__19_walk_forward_short_pulses_D__15p00_19p80__mirror.pkl b1264935f3f762e1a47c855e8b064741315522b337da9a4577298595039cd850
```

3. 四条 reference 的上游 X2 31DoF 源：

```text
/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_real_readonly_session04_canonical_31dof.npz
```

大小 `18,438,411 B`，SHA-256：

```text
287a134c9e8e260672f3b58c3968747604a5cb7e23af3186cfbba86ca4fc7
```

4. 至少一轮冻结四域 trace（建议只带 r1，r2 是逐项重复证据，可不重复搬运）：

```text
CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_ideal_teleop_r1_s260/
CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_filter_teleop_r1_s260/
CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_delay_teleop_r1_s260/
CWI_CrossEmbodiment_Sim/logs/x2_stage19_dcpeft_b0_stage152_b0_noise_teleop_r1_s260/
```

每条核心 JSONL 约 16 MB。它们让新服务器先做全离线前置审计，不必为了复现旧失败立即消耗 GPU。

5. 当前旧 X2 源树中被新仓库通过绝对路径引用、但尚未纳入 Git 的最小代码/配置子集。不要上传整个 69 GB 目录；应按 import 和 launcher 依赖打一个精简 source bundle，并保存 tree manifest。

### Docker 镜像缺口

百度网盘中已找到 SDK zip，但没有找到 `g1-deploy-dev:latest` Docker 镜像归档。若目标服务器无法重新获得 vendor base image，还必须执行：

```text
docker save g1-deploy-dev:latest | gzip > g1-deploy-dev_aimdk_x2_v1.tar.gz
```

并上传到：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/runtime/g1-deploy-dev_aimdk_x2_v1.tar.gz
```

否则只能运行 IsaacLab/直接 MuJoCo，不能复现官方 closed AimDK/ROS 域。

## 四、四种方法各自额外需要什么

| 阶段 | 额外资产 | 当前状态 |
|---|---|---|
| 前置审计 | forward4、metadata、Stage152 config、X2 MJCF | 除精确 seed bundle 外均有远端来源 |
| 方法1 OmniTrack式 | privileged obs/reward 配置、rollout 导出器、Stage152/WBT trainer | 代码在 Git/旧 X2 source；无需新外部软件 |
| 方法2 SBTO | `Atarilab/sbto` 源码及其精确 commit、许可证快照 | 当前百度归档未找到；取得代码前先确认许可证 |
| 方法3 DDR | matched 人体关键点/SMPL forward-walk，不是当前 X2 native forward4 | 当前尚无 matched 输入，需另选数据并建新 baseline |
| 方法4 DSMS | IPOPT、HSL MA57、实现源码/commit、许可证；可微/有限差分 MuJoCo 接口 | 当前 requirements 和网盘均未闭合；MA57 还涉及单独授权 |

不要一开始迁移全部 AMASS/PHUMA/BONES。前置审计和方法1/2只需要上述 4 条 reference；只有进入 DDR 或扩大 train/held-out 时才下载原始人体数据。

## 五、推荐下载顺序

1. Git clone 当前仓库。
2. 下载 SDK zip。
3. 下载待补传的 `x2_forward4_dynamics_feasibility_seed_v1.tar.gz`。
4. 下载 Phase55、Phase42、Phase30、Phase10。
5. 仅在继续 faithful WBT 时下载 Phase47/Phase36。
6. 仅在需要旧路径兼容时下载 `HUMAN+/x2_teleop_final.zip`。
7. 官方 closed 域需要时加载 Docker image tar。

示例：

```bash
bdpan download \
  'HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/official_x2_v1/sdk/aimdk-x2-v1.0.0-official.zip' \
  "$X2_WORKSPACE/downloads"
```

## 六、恢复后的首个判定

不要直接启动 OmniTrack 或长训。先验证：

1. checkpoint SHA 与 `b73c...0679` 一致；
2. forward4 的 5 个文件 SHA 与原 manifest 一致；
3. 旧 `baseline_report.md` 和 r1 trace 可离线复算；
4. 前置审计 1–5 全部生成；
5. 明确当前 reference 是 X2 native-derived，而非 GMR-derived。

只有上述五项通过，才决定先进方法1，还是因速度分布/镜像问题转向数据侧。
