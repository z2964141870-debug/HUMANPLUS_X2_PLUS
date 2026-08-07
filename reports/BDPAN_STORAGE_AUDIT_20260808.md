# bdpan-storage 上传链路审计（2026-08-08）

## 裁决

`baidu-netdisk/bdpan-storage` 可作为免 Cookie 的小文件管理补充，但不应替代 BaiduPCS-Go 承担 X2 checkpoint/数据集的无人值守大文件上传。

## 有价值的部分

- 使用官方 OAuth2；首次需浏览器授权，token 写入 `~/.config/bdpan/config.json`。
- 文档声明 token 接近过期时会自动刷新，通常不需要每天重导 Cookie。
- 可脚本调用 `bdpan upload <local> <remote> --json`，支持 Linux amd64/arm64 和独立 `--config-path`。

## 不能承担正式大包的原因

- 4 MB 分片但没有可靠的持久化断点续传；任一分片失败可能导致整文件重来。
- 公开 issue 仍报告 2.1 GB 慢上行稳定失败、分片 HTTP 500、21 GB 在 2048 分片处失败及目录上传挂起。
- 分片超时写死、单线程、无可配重试/超时。
- 远端限制在 `/apps/bdpan/`，不能直接写任意既有网盘目录。
- 仓库中只有 skill/安装器，核心 Go 客户端闭源，不便修补上传策略。

## 建议

1. 可做一次隔离 pilot：1 MB / 100 MB / 2 GB，不接正式 X2 大包。
2. 当前仍保留 `BaiduPCS-Go + 外层校验/重试` 为过渡路线，失效时不删本地包。
3. 真正稳定的无人值守大文件方案是百度开放 API：`precreate → superfile2 分片上传 → create`，自行持久化 uploadid/已传分片并控制重试；但需要百度开放平台应用和 OAuth 凭据。

## 官方 `baidu-netdisk/mcp` 补充审计

- 该仓库是更合适的二次开发基线：Python 源码明确实现 `precreate → 4 MB superfile2 分片 → create`，每片有 3 次指数退避重试。
- 原版仍不是可靠大包 CLI：未持久化 `uploadid`/已完成分片，进程重启后从头再来；`TIMEOUT` 和 `configure_session()` 定义了但未接入生成的 OpenAPI client 调用。
- 认证仅从 `BAIDU_NETDISK_ACCESS_TOKEN` 读取 access token，上传工具自身不保存 refresh token、不自动刷新。
- 个人用户目前是限时体验凭据，官方 README 警告测试密钥会不定期变更；正式接入指向企业开发者认证。
- 裁决：不安装整套 MCP 只为上传；从官方源码抽取上传协议，增加可原子落盘的状态文件、可配超时/重试、重启续传和最终 size/hash 校验，再做 100 MB→2 GB→正式包的递进门禁。

## 本轮安全边界

审计只在 `/tmp` 隔离 clone/安装探测；未读取用户 Cookie，未登录，未上传，未改用户网盘。
