# X2 大文件备份索引

## 2026-08-08 / task 30

- Git：`331fefe`，已推送 `origin/main`。
- 本地包：`x2_official_state_transport_task30_20260808.tar.gz`
- 大小：约 `19 MiB`。
- SHA256：`d15095dc0f005c6db6d5cc3a964684d937261aa38f31ce2af3dffd912e8758e3`
- 内容：Stage219/stand ONNX、Stage232 直行 6 次与左右转各 3 次官方 trace、Stage233 否定对照、直接 vendor-MJCF 延迟/预测关键 trace、内部 `SHA256SUMS`。
- 完整性：本地解包后 `sha256sum -c SHA256SUMS` 全部通过。
- 百度：上传重试 2 次均在传输前返回“获取用户 uk 错误/缺 STOKEN”；本地包完整，未误报上传成功。刷新一次 BDUSS+STOKEN 后可重传。

## 2026-08-08 / task 20

- Git：`8240aee`，已推送 `origin/main`。
- 本地包：`x2_official_stand_stop_task20_20260808.tar.gz`
- 大小：`3,751,767 bytes`
- SHA256：`18c38d46cab86d8caf0fb4ba125a9e9bf1537e3023fb293733b066c77952856a`
- 内容：stand i150 checkpoint/ONNX、两次官方 10 s 站立 trace、统一 stand→walk→stop trace 与报告。
- 百度：2026-08-08 上传失败，BaiduPCS-Go 返回缺少/失效 `STOKEN`；本地包完整，待刷新会话后重传。

## 2026-08-08 / task 10

- Git：`aa5324f`，已推送 `origin/main`。
- 本地包：`x2_stand_backend_task10_20260808.tar.gz`
- 大小：`3,338,499 bytes`
- 百度：同样因 `STOKEN` 失效尚未完成远端核验。
