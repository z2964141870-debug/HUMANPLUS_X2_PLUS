# 大文件存储与恢复规则

## 为什么不把权重直接提交到普通 Git

当前目录约 2.2 GB，其中主要是：

- `logs/`：约 1.8 GB；
- `checkpoints/`：约 330 MB；
- 源码、文档、配置和报告：远小于上述两项。

Git 会永久保留历史对象。即使之后删除大权重，仓库历史仍可能继续占空间。
Git LFS 虽适合模型文件，但首次 `git add` 通常还会在 `.git/lfs/objects` 保存
本地对象副本，在磁盘紧张时不应直接对现有约 1.1 GB 保留权重启用。

## 推荐的两层存储

```text
Git repository
├── src / scripts / tools / tests
├── configs / hooks
├── Markdown 结果卡与决策台账
├── compact reports / manifests
└── 大文件的 SHA256、大小、来源与恢复说明

External artifact storage
├── retained baseline checkpoint
├── retained best/research checkpoints
├── 必须保留的原始 trace
└── 视频与数据集归档
```

外部存储可以是私有对象存储、网盘、实验室 NAS、私有 Hugging Face 数据集，
或明确配额后的 Git LFS。上传并验证 SHA256 后，才允许删除本机副本。

## 安全删除门

任何大文件只有同时满足以下条件才允许从本机清理：

1. 已确认属于本项目新生成文件，而不是旧工程资产；
2. 路径、字节数和 SHA256 已写入 `manifests/`；
3. 远端上传完成；
4. 从远端重新下载后 SHA256 一致；
5. 对应实验指标、配置和结果卡已进入 Git；
6. baseline、best、final、首个明确失败等保留策略已执行。

在远端未验证前，Git 初始化本身不会释放当前 2.2 GB，也不授权删除任何权重。

## 推送步骤

远端仓库建立后：

```bash
git remote add origin <你的私有仓库地址>
git push -u origin main
```

大文件不要绕过 `.gitignore` 使用 `git add -f`。需要备份权重时，应先选定外部
artifact store，再按 manifest 逐个上传与回读校验。

