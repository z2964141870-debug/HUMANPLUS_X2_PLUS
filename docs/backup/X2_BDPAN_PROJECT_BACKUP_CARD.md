# X2 项目官方 bdpan 备份卡片

> 版本：v1.0
> 日期：2026-08-08
> 适用：`/home/humanplus/projects/ZHY` 下的 X2 代码外大文件、checkpoint、ONNX、视频和归档包

## 当前结论

官方 `bdpan` 已经能够服务当前 X2 项目的上传归档需求：

- OAuth 登录成功，Token 约 30 天有效并支持自动刷新，不需要每天导入 Cookie；
- 32 MiB 随机文件上传成功；
- 当前项目最大 checkpoint `stage152_B_dual_equal_split_init.pt`（193,683,894 字节）约 29 秒上传成功；
- 远端文件名和字节数核验一致；
- 项目封装脚本支持有限重试、并发锁、同名保护、重复运行跳过和本地 SHA-256 manifest。

当前边界也必须保留：

- 远端只能写入“我的应用数据/bdpan/”；
- `bdpan download` 在本机多次遇到百度 `/apps` 列表接口 `EOF`；
- 因此当前自动验收只能证明远端路径和字节数一致，不能声称已经完成下载回读 SHA-256；
- 对当前最大约 194 MB 的权重已经实测够用，但尚未证明数 GB 单文件可靠。

## 固定入口

```bash
BDPAN="/home/humanplus/.local/bin/bdpan"
UPLOAD="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/tools/backup/x2_bdpan_upload_resilient.sh"
```

安装版本：`bdpan 3.8.5`。

认证配置位于 `~/.config/bdpan/config.json`。禁止读取、打印、复制或提交该文件。

## 登录检查

```bash
bdpan whoami
```

未登录或刷新失败时，只能运行官方登录脚本：

```bash
bash "/home/humanplus/projects/Human+智能服装动作捕捉系统/tools/bdpan-storage/skills/baidu-drive/scripts/login.sh"
```

不要直接把密码、Cookie 或 Token 传给 agent。OAuth 页面返回的授权码只通过登录脚本标准输入提交。

## 上传单个 X2 文件

```bash
bash "$UPLOAD" \
  /home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/checkpoints/model.pt \
  x2-project-backups/2026-08-08
```

上述远端目录实际显示为：

```text
我的应用数据/bdpan/x2-project-backups/2026-08-08/
```

脚本行为：

1. 验证 OAuth 登录；
2. 计算本地字节数和 SHA-256；
3. 使用 `flock` 防止重复上传；
4. 上传失败最多尝试 4 次；
5. 上传后最多进行 3 次远端列表核验；
6. 同名同大小文件自动跳过，同名不同大小则拒绝覆盖；
7. 在 `docs/backup/manifests/` 生成 JSON 清单。

调整尝试次数：

```bash
BDPAN_UPLOAD_ATTEMPTS=6 bash "$UPLOAD" LOCAL_FILE REMOTE_DIRECTORY
```

## 与 Git 的配合

每 10 个实质任务执行一次：

```text
代码、配置、脚本、Markdown
        → Git commit + GitHub push

checkpoint、ONNX、大视频、归档包
        → x2_bdpan_upload_resilient.sh

Git commit、文件路径、大小、SHA-256、网盘相对路径
        → backup manifest
```

不要把 `.pt`、大型 `.onnx`、数据集或视频提交到 Git。

## 当前已验证的项目文件

| 文件 | 字节数 | 结果 |
|---|---:|---|
| `stage306_transition_head_model_2657.pt` | 1,222,965 | 上传成功，远端大小一致 |
| `stage306_s2657_transition_head_actor.onnx` | 329,334 | 封装脚本上传成功，重复运行正确跳过 |
| `stage152_B_dual_equal_split_init.pt` | 193,683,894 | 上传成功，远端大小一致 |

远端目录：

[打开 X2 2026-08-08 归档](https://pan.baidu.com/disk/main#/index?category=all&path=%2Fapps%2Fbdpan%2Fx2-project-backups%2F2026-08-08)

## 恢复策略

在官方 CLI 下载 `EOF` 修复前：

1. 日常自动上传使用 `bdpan`；
2. 需要恢复时优先使用百度网盘桌面客户端，从“我的应用数据/bdpan”下载；
3. 下载后必须与本地/Git manifest 中的 SHA-256 比较；
4. 若桌面客户端不可用，再刷新 BaiduPCS-Go Cookie 作为回退通道。

只有下载回读 SHA-256 一致，才能称为端到端完整性验证；远端字节数一致只能称为上传存在性验证。
