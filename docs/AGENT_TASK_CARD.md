# Agent 开工任务卡：X2 / 机器人项目通用规范

版本：`v1.1`
生效日期：`2026-08-15`
适用范围：`/home/yu/projects` 下的项目，以及关联的环境、数据、实验、远程计算机和归档。

> 这是交给任何后续 agent 的强制开工说明。agent 必须先阅读本卡，再开始写入、运行长实验或连接真实机器人。若规则与用户当次明确指令冲突，先报告冲突和风险，不得静默覆盖历史资产。

## 0. 开工报告必须先回答

在执行第一条会改变状态的命令前，先在回复/日志中写明：

```text
项目根目录：
代码/配置写入目录：
数据/日志/checkpoint 写入目录：
环境目录：
当前机器/hostname：
Git 仓库、分支、commit、工作树状态：
本次模式：只读 / 仿真 / sim2sim / 真实硬件
是否涉及 Orin、X2 三台 PC、BLE 衣服或机器人：
预计新增文件和预计磁盘占用：
```

先执行并记录（路径按实际项目替换）：

```bash
hostname
pwd
df -h /home /media/yu/FAFF-E977 2>/dev/null || true
git -C /home/yu/projects/<project> status --short --branch
git -C /home/yu/projects/<project> rev-parse HEAD
git -C /home/yu/projects/<project> remote -v
```

如果目录不是 Git 仓库，必须明确写“不是 Git 仓库”，不要自行初始化或改变目录归属。

## 1. 设备与远程安全边界

1. `hp3090` 默认只代表远程 3090 计算机，默认只能做代码检查、仿真、离线评估和训练；不得把它当成机器人控制机。
2. 未获得用户当次明确授权时，不得连接或操作 Orin、X2 三台 PC、真实机器人、BLE 衣服/裤子、遥操桥接服务、电机、上电/下电流程。
3. 仿真、sim2sim、真实硬件必须分开记录，不能用仿真成功推断真机安全或效果。
4. 任何真实硬件操作都必须先报告：目标设备、当前状态、预期动作、停止/急停方式和回滚方式；默认先做只读检查或空跑。
5. 不执行 `rm -rf`、`git reset --hard`、`git clean -fd`、覆盖式复制、公共历史重写或 `git push --force`，除非用户明确给出精确目标和授权。
6. 长任务使用 `tmux`/`nohup` 前记录 PID、工作目录、完整命令和日志路径；只终止自己启动的 PID，不杀未知进程。
7. 远程命令、上传和下载使用绝对路径；先 `ls/stat/sha256sum` 确认目标，再写入。

## 2. 目录、磁盘和环境规范

### 2.1 代码、配置、脚本和文档

统一放在：

```text
/home/yu/projects/<project>/
```

允许放入源码、配置、启动脚本、测试、文档、精简 manifest、依赖声明和小型示例数据。旧路径如仍被脚本使用，只保留指向新目录的软链接；不得保留两份可修改代码。

第三方只读源码放在：

```text
/home/yu/projects/external/<name>/
```

或项目明确的 `third_party/`；未经授权不修改、不提交其源码。

### 2.2 环境、依赖和编译产物

项目专属环境优先放在：

```text
/home/yu/env/<project>/
```

也可使用已验证的全局环境，但必须记录环境名称、Python/torch/CUDA/ONNX Runtime 版本。环境目录、`.venv`、conda prefix、runtime、编译缓存不得提交 GitHub 或上传百度网盘。提交 `requirements*.txt`、`environment*.yml`、lock 文件和安装说明即可。

不要直接移动 `/home/yu/miniconda3`；如确需迁移，先导出环境并验证所有 prefix、import 和启动脚本。

### 2.3 大数据、模型、日志和结果

统一放在：

```text
/media/yu/FAFF-E977/data/<project>/
```

推荐子目录：

```text
raw/          原始数据，只读
processed/    处理后数据
checkpoints/  模型权重
logs/         训练、评估和运行日志
cache/        可重建缓存
archives/     压缩包和迁移快照
manifests/    SHA256、大小、来源、配置、commit 和恢复记录
```

大文件必须直接写入 data mount，不要先写进 Git 工作树再搬运。项目目录只保留代码、配置、测试、文档、精简 manifest 和相对路径/软链接。若 `/media/yu/FAFF-E977` 未挂载，先停止大文件写入并报告，不要悄悄把数据堆到 `/home`。

### 2.4 磁盘检查与清理

开始长实验前检查 `df -h`，必要时检查 `du -sh`。不得删除原始数据、checkpoint、失败样本、旧报告或历史日志。清理前必须同时具备：来源确认、文件大小、SHA256、备份/移动目标、恢复说明，并确认没有进程正在使用。

可重建 cache 也要先确认无运行进程；删除后在报告中列出删除范围和释放空间。优先采用“移动到 `archives/` + 软链接”而不是覆盖或删除。

## 3. Git / GitHub 版本管理

### 3.1 开工检查

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
git remote -v
```

先记录工作树是否 dirty。已有的 dirty/untracked 文件属于用户资产；与本任务无关的文件不得删除、格式化、移动或加入提交。若改动范围重叠，停止并报告。

### 3.2 分支、提交和推送

- `main` 是稳定分支；工作使用 `work/<agent>-<topic>`。
- 一项任务尽量一个原子提交，格式建议：`<area>: <change>`，例如 `teleop: fix BLE UTC pairing`。
- 只提交源码、配置、脚本、测试、文档、依赖声明和精简 manifest。
- 禁止提交环境、checkpoint、训练日志、视频、数据集、模型缓存、大压缩包、个人绝对路径、密码、Cookie、token、私钥、设备隐私信息。
- 用 `.gitignore` 排除 `logs/`、`checkpoints/`、`cache/`、`.venv/` 等输出，但不要为掩盖已有大文件而单方面删除历史跟踪内容。
- 推送前运行与本任务相关的最小测试，并记录命令、结果、未解决风险；只推送用户授权的远端和分支。
- 推送后用 `git ls-remote` 或远端查看确认 commit；不 force-push、不重写公共历史。
- 冻结版本用 `vYYYY.MM.DD[-N]` 标签，并在 manifest 记录 commit、工作树、依赖和数据 SHA256。

### 3.3 提交前检查

```bash
git diff --check
git status --short
git diff --stat
git diff -- <only-files-for-this-task>
```

提交说明必须列出：改动文件、测试命令、结果、数据产物位置和已知限制。

## 4. 百度网盘备份规范

百度网盘只用于外部 artifact 备份，不替代 GitHub、工作目录或 data mount。

上传前：

```bash
sha256sum artifact.tar.zst > artifact.tar.zst.sha256
stat -c '%s %n' artifact.tar.zst artifact.tar.zst.sha256
```

远端路径统一使用：

```text
/apps/bdpan/X2/<project>/<YYYY-MM-DD>/<artifact>
```

优先使用项目已有的 `tools/backup/x2_bdpan_upload_resilient.sh`（参考实现位于 `ZHY/CWI_CrossEmbodiment_Sim`）。上传前检查登录状态、远端目录和并发锁；上传后记录远端路径、远端字节数、上传时间、本地 SHA256、Git commit 和 manifest。百度接口通常不能回读 SHA256，至少验证远端文件名/字节数，必要时下载回本地再校验。

远端未验证前不得删除本地 artifact。禁止上传环境、密钥、Cookie、未脱敏日志或未经授权的第三方数据。授权只能通过官方登录流程和交互式 stdin 完成，禁止把密码、Cookie、STOKEN 写入命令参数、脚本、任务卡或日志。

## 5. 实验与数据记录

每个实验开始前写清：

```text
假设/目的：
输入数据及来源：
模型/权重及 SHA256：
代码 commit：
环境与硬件：
固定配置、随机种子和时间范围：
成功判据：
安全边界：
```

执行规范：

1. 先做 import/模型加载/短时 smoke test，再启动长实验。
2. 一次只改变一个主要变量；失败也要保存并标注 `passed`、`failed` 或 `blocked`。
3. 日志写入 `/media/yu/FAFF-E977/data/<project>/logs/`，结果写入 `processed/`，权重写入 `checkpoints/`，原始输入保持 `raw/` 只读。
4. 每个结果保存命令、commit、环境版本、输入/输出 SHA256、耗时、GPU/显存、异常和复现命令。
5. 不把失败样本“清理掉”；失败案例是后续诊断和论文实验的一部分。

## 6. 结束时必须交接

```text
结论/当前状态：
项目根目录：
代码、数据、环境实际写入位置：
机器与硬件安全状态：
Git 分支/commit/工作树：
改动文件：
测试命令与结果：
数据产物、大小和 SHA256：
已知问题/风险：
下一步可直接执行的命令：
恢复/回滚路径：
```

不得只说“完成”或“文件已存在”；必须让下一位 agent 能按报告复现或恢复。

## 7. 开工/收工清单

```text
[ ] 已阅读本卡并报告项目根目录、写入目录、机器和 Git 状态
[ ] 已确认本次是只读、仿真、sim2sim 还是真实硬件
[ ] 已确认大数据写入 /media/yu/FAFF-E977/data
[ ] 已确认环境写入 /home/yu/env 或使用既有已验证环境
[ ] 已检查磁盘空间和运行进程
[ ] 已确认不会提交 checkpoint、日志、环境、密钥或个人路径
[ ] 已保留并记录已有 dirty/untracked 用户文件
[ ] 若上传百度：已生成 SHA256、大小和远端 manifest
[ ] 已运行最小验证并保存日志/结果
[ ] 已提交原子 Git commit（如任务要求）并记录推送结果
[ ] 已填写结束交接报告和恢复路径
```

## 8. canonical 位置

- 全局开工卡：`/home/yu/projects/AGENT_TASK_CARD.md`
- BFM-Zero 版本化副本：`/home/yu/projects/BFM-Zero/docs/AGENT_TASK_CARD.md`
- 本卡升级或替换旧版前，旧版应保留为带日期的备份，不得直接丢弃。
