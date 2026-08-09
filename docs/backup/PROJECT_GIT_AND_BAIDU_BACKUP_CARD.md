# 项目版本管理与大文件归档主卡片

> 版本：v2.0
> 日期：2026-08-08
> 优先级：后续项目首先阅读本卡片
> 适用范围：代码、配置、报告、实验日志、数据集、checkpoint、ONNX、视频和压缩包的长期管理

## 一句话规则

```text
代码、配置、脚本、Markdown
        → Git / GitHub

checkpoint、数据集、长日志、视频、压缩包
        → 官方 bdpan / 百度网盘

恢复下载、端到端 SHA-256 抽检、普通网盘目录兼容
        → 百度网盘桌面客户端或 BaiduPCS-Go
```

Git 负责可追溯的版本历史，百度网盘负责 Git 不适合承载的大文件。两边必须通过同一份 manifest 中的 Git commit、路径、字节数和 SHA-256 相互关联。

## 1. 文件应该放在哪里

| 文件类型 | 推荐位置 | 说明 |
|---|---|---|
| 源代码、配置、脚本、Markdown | GitHub | 应提交、评审并保留历史 |
| URDF/MJCF、小型 mesh 和测试资产 | GitHub | 单文件尽量低于 50 MB |
| checkpoint、模型权重、较大 ONNX | 百度网盘 | 不提交 Git |
| 原始或处理中间数据集 | 百度网盘 | Git 只保存来源、版本和转换脚本 |
| 大型 JSONL、长日志、视频、压缩包 | 百度网盘 | Git 保存摘要与远端索引 |
| token、Cookie、私钥、账号配置 | 都不能上传 | 只保存在本机受限目录 |

保守门槛：单文件超过 50 MB 时优先放百度网盘，不要等到 GitHub 拒绝才处理。

## 2. 新项目 Git 初始化

```bash
git init -b main
git config user.name "z2964141870-debug"
git config user.email "z2964141870-debug@users.noreply.github.com"
```

`.gitignore` 至少包含：

```gitignore
# Python / editors
__pycache__/
*.py[cod]
.pytest_cache/
.vscode/
.idea/

# Secrets and local credentials
.env
.env.*
!.env.example
*.pem
*.key
.baidupcs_go/
.baidu_bypy/

# Large experiment artifacts
checkpoints/
logs/
videos/
datasets/
*.pt
*.pth
*.ckpt
*.onnx
*.tar
*.tar.gz
*.zip
*.uploading
```

如确实需要跟踪某个小型测试权重，应使用精确的 `!path/to/file` 例外，不能整体开放 `*.pt` 或 `*.onnx`。

首次提交前检查：

```bash
git status --short
git add -A
git diff --cached --stat
git diff --cached --check
git commit -m "chore: initialize project archive"
```

确认 staged payload 没有异常大文件：

```bash
git ls-files -s \
  | awk '{print $2}' \
  | git cat-file --batch-check='%(objectsize) %(rest)' \
  | sort -nr \
  | head -20
```

不要提交密码、Cookie、Token、SSH 私钥或包含这些内容的调试日志。

## 3. 连接并推送 GitHub

先在 GitHub 创建空仓库，再执行：

```bash
git remote add origin git@github.com:OWNER/REPOSITORY.git
git push -u origin main
```

本机专用 SSH Host alias：

```bash
git remote add origin \
  git@github-humanplus-x2:OWNER/REPOSITORY.git
git push -u origin main
```

新的仓库必须在 `Settings → Deploy keys` 中授权对应公钥并勾选写权限，或配置账号级 SSH key；绝不上传或展示私钥。

推送后核验：

```bash
git rev-parse HEAD
git ls-remote origin refs/heads/main
git status --short --branch
```

本地 `HEAD` 与远端 `main` 哈希必须一致，工作区是否干净也必须如实报告。

里程碑可以增加标签：

```bash
git tag -a stage-01 -m "Stage 01 verified"
git push origin stage-01
```

checkpoint 不随标签进入 Git；manifest 必须记录对应 commit 和百度网盘路径。

## 4. 官方 bdpan：默认上传通道

当前安装：

```bash
BDPAN="/home/humanplus/.local/bin/bdpan"
"$BDPAN" version
"$BDPAN" whoami
```

当前实测版本为 `bdpan 3.8.5`。它使用 OAuth 2.0，Token 约 30 天有效并支持自动刷新，通常不需要每天重新导入 Cookie。

认证配置位于：

```text
~/.config/bdpan/config.json
```

禁止读取、打印、复制、上传或提交该文件。

未登录或自动刷新失败时，只能运行官方登录脚本：

```bash
bash "/home/humanplus/projects/Human+智能服装动作捕捉系统/tools/bdpan-storage/skills/baidu-drive/scripts/login.sh"
```

不要发送百度密码、Cookie 或 Token。OAuth 页面返回的授权码只通过官方登录脚本的标准输入提交。

### 权限边界

官方 `bdpan` 只能操作：

```text
我的应用数据/bdpan/
```

它不能直接操作普通网盘根目录。我们已经在其允许范围内建立同名项目根：

```text
我的应用数据/bdpan/HUMAN+/
```

命令参数使用相对于 `/apps/bdpan/` 的路径，不要写 `/apps/bdpan/` 前缀。

## 5. 百度网盘目录契约

所有新项目统一使用：

```text
HUMAN+/PROJECT_NAME/YYYY-MM-DD/
├── checkpoints/
├── models/
├── datasets/
├── reports/
├── logs/
├── videos/
└── manifests/
```

日期表示归档批次，不一定是模型训练日期。不要把多个项目混进同一远端目录。

X2 当前正式目录：

```text
HUMAN+/HUMANPLUS_X2_PLUS/YYYY-MM-DD/
```

## 6. 项目安全上传脚本

X2 已验证入口：

```bash
UPLOAD="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/tools/backup/x2_bdpan_upload_resilient.sh"
```

上传 checkpoint：

```bash
bash "$UPLOAD" \
  /absolute/path/model_step_001000.pt \
  'HUMAN+/HUMANPLUS_X2_PLUS/2026-08-08/checkpoints'
```

上传 ONNX：

```bash
bash "$UPLOAD" \
  /absolute/path/policy.onnx \
  'HUMAN+/HUMANPLUS_X2_PLUS/2026-08-08/models'
```

脚本保证：

1. 验证 OAuth 登录；
2. 计算本地字节数和 SHA-256；
3. 使用 `flock` 防止多个 agent 重复上传；
4. 上传失败有限重试，默认最多 4 次；
5. 百度返回异常 JSON 时拒绝上传，不冒险覆盖；
6. 默认以 `bdpan upload` 返回成功为本次完成，不逐文件做上传后轮询；
7. 同名同大小文件自动跳过；
8. 同名不同大小文件拒绝覆盖；
9. 在 Git 仓库的 `docs/backup/manifests/` 生成 JSON 清单，并明确标记为待每日核验；
10. 无论成功失败都不删除本地源文件。

调整尝试次数：

```bash
BDPAN_UPLOAD_ATTEMPTS=6 bash "$UPLOAD" LOCAL_FILE REMOTE_DIRECTORY
```

只有用户明确要求当日统一检查时，才启用上传后远端字节核验：

```bash
BDPAN_VERIFY_AFTER_UPLOAD=1 bash "$UPLOAD" LOCAL_FILE REMOTE_DIRECTORY
```

若文件已经上传，同名同大小的预检会直接生成核验清单，不会重复上传。默认不做逐文件上传后核验，以减少百度接口轮询和等待时间。

其他项目可以复用该脚本，但应把脚本复制进自己的 Git 仓库并将路径写入本项目主卡或 README。

## 7. 训练自动归档约定

训练中的文件使用临时后缀：

```text
model_step_001000.pt.uploading
```

文件完全写入并关闭后，再原子改名：

```bash
mv model_step_001000.pt.uploading model_step_001000.pt
```

上传脚本只能匹配完成态文件，不上传 `.uploading`、临时目录或仍在写入的日志。

默认每 10 个实质任务执行一次版本归档：

```text
代码、配置、报告
        → Git commit + GitHub push

checkpoint、ONNX、大视频、压缩包
        → 官方 bdpan

Git commit、路径、大小、SHA-256、远端位置
        → manifest
```

明显里程碑、关键权重或破坏性修改前后，可以提前备份，不必机械等待第 10 个任务。

## 8. Manifest 与验收

每批归档至少记录：

- Git commit；
- 本地相对路径或明确绝对路径；
- 文件字节数；
- SHA-256；
- 远端相对路径；
- 数据来源或实验阶段；
- 上传日期；
- 验证范围。

本地计算：

```bash
stat -c '%s %n' /absolute/path/file
sha256sum /absolute/path/file
```

默认上传与验收分开执行：上传任务只要求官方 CLI 返回成功；远端检查由用户每天明确触发一次，集中完成。

验收分为两级：

### 一级：每日存在性验证（不随每次上传自动执行）

- 远端路径正确；
- 远端字节数与本地一致；
- manifest 已生成并进入 Git 或上传到 `manifests/`。

一级验收不能证明远端内容逐字节一致。

### 二级：端到端完整性验证

- 从百度网盘重新下载关键文件；
- 对下载副本计算 SHA-256；
- 与 manifest 完全一致。

只有二级验收通过，才能称为完整备份闭环。

百度显示的远端 MD5 可能是分片语义或异常字符串，不能替代下载回读 SHA-256。

## 9. 恢复文件

当前 `bdpan download` 在本机多次遇到百度 `/apps` 列表接口 `EOF`。在官方 CLI 修复前，恢复优先级为：

1. 百度网盘桌面客户端，从“我的应用数据/bdpan/HUMAN+”下载；
2. 下载后依据 Git manifest 执行 `sha256sum`；
3. 桌面客户端不可用时，使用 BaiduPCS-Go 兼容通道；
4. 校验通过后才投入训练、评估或部署。

关键里程碑仍建议由用户择日执行 SHA-256 回读抽检，但不在每次上传后自动执行。

## 10. BaiduPCS-Go：回退与普通目录通道

当前入口：

```bash
PCS="/home/humanplus/projects/ZHY/baidupcs"
"$PCS" --version
"$PCS" who
"$PCS" quota
```

版本：`BaiduPCS-Go v4.0.1`。

共享配置：

```text
/home/humanplus/projects/ZHY/.baidupcs_go/bin/
/home/humanplus/projects/ZHY/.baidupcs_go/config/
```

安全要求：

- `.baidupcs_go/` 必须进入 `.gitignore`；
- 配置目录权限为 `700`，配置文件权限为 `600`；
- Cookie、BDUSS、STOKEN 和密码不得发到聊天、报告或 Git；
- Cookie 失效时需要用户重新导入，agent 不得提取浏览器 Cookie。

BaiduPCS-Go 可操作普通网盘 `/HUMAN+/...`，也作为官方 bdpan 下载异常时的回退方案：

```bash
"$PCS" upload \
  /absolute/path/model.pt \
  "/HUMAN+/PROJECT_NAME/YYYY-MM-DD/checkpoints" \
  --norapid --policy rsync
```

```bash
"$PCS" download \
  "/HUMAN+/PROJECT_NAME/YYYY-MM-DD/checkpoints/model.pt" \
  --saveto /tmp/PROJECT_NAME_verify
```

`--norapid` 在本机比秒传接口稳定；`--policy rsync` 会跳过远端同名同大小文件。网络错误时保留本地文件并有限重试。

## 11. 已知失败路线与边界

### bypy

本机 `bypy v1.8.9` 的 OAuth 和小文件上传可用，但约 194 MB checkpoint 曾出现：

```text
Error 31064
Slice MD5 mismatch
```

缩小分片、强制重算哈希和绕过代理均未解决；Python 3.13 多进程 `syncup` 也出现退出异常。因此 bypy 只保留为 `/apps/bypy/` 小文件兼容链路，不承担正式大文件归档。

### 官方 bdpan

- 上传当前项目最大约 194 MB checkpoint 已成功；
- 32 MiB 随机文件与小型 PT/ONNX 均成功；
- 尚未证明数 GB 单文件可靠；
- 下载存在间歇性 `/apps` 列表 `EOF`；
- 只能操作“我的应用数据/bdpan/”。

### BaiduPCS-Go

- 能操作普通网盘目录并完成历史端到端回读；
- 大文件上传比 bypy 可靠；
- Cookie 可能失效，需要用户重新导入；
- 因此保留为恢复和兼容通道，而不是默认每日自动上传入口。

## 12. 新项目最短任务清单

- [ ] 新建独立 GitHub 仓库。
- [ ] 初始化 Git、`.gitignore` 和提交身份。
- [ ] 确认 secrets 与大文件未进入 staged payload。
- [ ] 首次 commit 并推送，核对本地与远端 commit。
- [ ] 在 bdpan 中建立 `HUMAN+/PROJECT_NAME/YYYY-MM-DD/` 分类目录。
- [ ] 将安全上传脚本纳入项目 Git。
- [ ] 生成包含 Git commit、大小、SHA-256 和远端路径的 manifest。
- [ ] 使用官方 bdpan 上传完成态大文件。
- [ ] 用户要求时，每日集中核对新增远端路径和字节数。
- [ ] 将 manifest 提交 Git，并上传一份到网盘 `manifests/`。
- [ ] 用桌面客户端或 BaiduPCS-Go 下载一个关键文件做 SHA-256 抽检。
- [ ] 只清理明确的临时探针；不自动删除源码、数据集或 checkpoint。
- [ ] 在项目 README 中记录 GitHub 仓库和百度网盘归档路径。

## 13. 当前 X2 已验证实例

GitHub：

```text
https://github.com/z2964141870-debug/HUMANPLUS_X2_PLUS
```

官方 bdpan 正式目录：

```text
我的应用数据/bdpan/HUMAN+/HUMANPLUS_X2_PLUS/YYYY-MM-DD/
```

已验证上传：

| 文件 | 字节数 | 结果 |
|---|---:|---|
| `stage306_transition_head_model_2657.pt` | 1,222,965 | 上传成功，远端大小一致 |
| `stage306_s2657_transition_head_actor.onnx` | 329,334 | 上传成功，重复运行正确跳过 |
| `stage152_B_dual_equal_split_init.pt` | 193,683,894 | 上传成功，远端大小一致 |

早期三份验证文件仍在：

```text
我的应用数据/bdpan/x2-project-backups/2026-08-08/
```

后续新文件统一进入 `HUMAN+/HUMANPLUS_X2_PLUS/`。管理卡已上传到该项目的 `manifests/`。

历史 BaiduPCS-Go 归档：

```text
/HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/
```

该历史归档包含 6 个 checkpoint、2 个 trace 和 1 份 manifest，总计 1,268,680,543 字节；远端字节数均与本地一致，并完成一份约 91 MB trace 的下载回读 SHA-256 抽检。
