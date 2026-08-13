# X2 新 RTX 3090 Linux 工作站：百度网盘迁移入口

版本：2026-08-12

本目录让不了解百度网盘的 AI Agent 也能完成迁移。迁移只使用官方 `bdpan`
CLI；不复制老机器的 Cookie、Token、百度密码或 SSH 私钥。

## 0. 真实性边界

- 当前 Git 基线 HEAD：`ac4d34f97a461b3e5c61bbec754ddef5f89449da`。
- Cycle20–40 和 Phase40 WIP 尚未进入 GitHub，必须下载本清单中的周期包与
  `x2_current_worktree_delta_phase40_20260812.tar.gz`。
- `full` 清单约 7.1 GiB，是可复现实验集合，不是老机器 123 GiB 的缓存镜像。
- Conda、CUDA、Isaac Sim 和 NVIDIA 驱动必须在新机重建，禁止复制旧环境目录。
- `bdpan` 的 OAuth 配置只保存在新机 `~/.config/bdpan/config.json`，严禁打印或上传。

## 1. 首次取得本引导包

如果新机尚无 `bdpan`，先从百度官方 CDN 安装 Linux x86-64 版本。也可以让用户
通过百度网盘网页下载本引导包，再运行包内 `install_local_bdpan.sh`。

本引导包的远端路径：

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/new_pc_migration/
  x2_new_pc_bdpan_bootstrap_20260812.tar.gz
```

## 2. 一次性 OAuth 登录

```bash
bash login_bdpan.sh
```

脚本会输出百度授权链接。Agent 必须把链接原样交给用户；用户在自己的浏览器中
授权并把 32 位授权码粘贴回终端。授权码通过 stdin 提交，不得放入命令参数、日志
或聊天记录。Token 正常可自动刷新，不需要每天导入 Cookie。

## 3. 下载全部可复现集合

默认下载 `core + full`：

```bash
export X2_MIGRATION_ROOT="$HOME/x2_migration_20260812"
bash download_x2_migration.sh --full
```

脚本逐文件检查字节数和 SHA-256；已正确下载的文件自动跳过，失败后重复运行同一
命令即可。不要跳过 SHA。

## 4. 恢复代码和最小运行资产

先预览：

```bash
bash restore_x2_core.sh
```

确认目标路径后执行：

```bash
bash restore_x2_core.sh --execute
```

默认布局保持历史绝对路径：

```text
$HOME/projects/ZHY/CWI_CrossEmbodiment_Sim
$HOME/projects/ZHY/x2_official_rl_deploy_v1
$HOME/projects/ZHY/dsms_workspace
$HOME/x2_teleop_final
$HOME/humanoid-GPT/Humanoid-GPT
$HOME/projects/Human+智能服装动作捕捉系统/general_motion_retargeting
```

若新机用户名不是 `humanplus`，建议创建同名用户或后续统一修正 launcher 中的绝对
路径。不要通过覆盖 `/home` 的方式强行恢复。

## 5. Docker runtime

```bash
cat "$X2_MIGRATION_ROOT/downloads/runtime/"x2_aimdk_runtime_images_v1.tar.gz.part-* \
  | gzip -dc | docker load
docker image inspect x2-aimdk-humble:1.0 g1-deploy-dev:latest
```

## 6. Python/Isaac 环境

按下载后仓库中的以下文件重建，不复制旧 Conda：

```text
docs/backup/X2_MIGRATION_FINAL_REQUIREMENTS_20260810.md
requirements.txt
```

关键锁：Python 3.11.15、Isaac Sim 5.1.0.0、IsaacLab
`37ddf626871758333d6ed89cf64ad702aef127d0`、PyTorch 2.8.0+cu128、
MuJoCo 3.3.7。

## 7. 首次验收

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import mujoco; print(mujoco.__version__)'
docker image inspect x2-aimdk-humble:1.0 g1-deploy-dev:latest

cd "$HOME/projects/ZHY/CWI_CrossEmbodiment_Sim"
PYTHONPATH="$PWD:$PWD/src:$PWD/tools" \
  python -m pytest -q tests/test_phase40_privileged_generator_live_wiring.py
```

Phase40 当前是 WIP：静态测试通过，但 live-zero 仍停在 Hydra wrapper config 接线，
不得写成训练已解锁。迁移后的第一项工作应先完成 live-zero，不直接长训。
