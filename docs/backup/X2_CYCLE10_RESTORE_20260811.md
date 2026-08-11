# X2 第10任务周期恢复说明（2026-08-11）

本归档覆盖上一百度大包之后的10个实质任务：动态重定向Phase10–17与BASE Phase38–39。

## 内容

- `HUMANPLUS_X2_PLUS_cycle10.bundle`：完整 Git 仓库历史，截止归档内容commit。
- `external_phase19_v2/`：Phase19 r2–r5 四条stateful-v2 sidecar；Phase38/39的连续成功suffix与支持域审计依赖它们。
- `X2_CYCLE10_RESTORE_20260811.md`：本说明。

上一周期的大型动态重定向缓存仍位于：

`HUMAN+/HUMANPLUS_X2_PLUS/2026-08-11/dynamic_retargeting/x2_dynamic_retargeting_race_full_20260811.tar.gz`

本周期不重复打包旧mmap、Docker镜像、official SDK和迁移源代码包。

## 恢复 Git

```bash
git clone HUMANPLUS_X2_PLUS_cycle10.bundle CWI_CrossEmbodiment_Sim
cd CWI_CrossEmbodiment_Sim
git switch main
```

## 恢复 Phase19 外部证据

把四个sidecar放回：

```text
/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807/
```

文件名保持不变。随后可运行：

```bash
PYTHONPATH=tools python tools/official_x2/audit_phase38_sequence_consistent_bridge_target.py \
  --scene /path/to/official/scene.xml \
  --output /tmp/phase38.json
```

## 真实性边界

- 百度上传成功只表示upload命令返回成功；远端大小/SHA回读按项目约定由每日人工审计完成。
- Git bundle不包含未跟踪文件。
- official SDK、Humanoid-GPT、PHUMA和既有迁移包沿用此前独立归档，不在本包重复。
