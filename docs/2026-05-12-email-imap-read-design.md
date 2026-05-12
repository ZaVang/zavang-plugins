# Email IMAP 读取功能设计

## 目标

为 email-sender 插件增加 IMAP 收件能力，让 Claude Code 可以读取邮箱中的邮件。

## 文件变更

```
plugins/email-sender/
├── .claude-plugin/plugin.json   # bump version 1.0.0 → 1.1.0, update keywords
├── scripts/
│   ├── send_email.py            # 不改
│   └── check_email.py           # 新增
└── skills/
    ├── send-email/SKILL.md      # 不改
    └── check-email/SKILL.md     # 新增
```

## 架构

单入口 `check_email.py`，4 个子命令：`list | search | read | monitor`。

纯 Python stdlib — `imaplib` + `email`，零新依赖。

## 子命令

| 命令 | 参数 | 功能 |
|------|------|------|
| `list` | `--limit 10`, `--folder INBOX`, `--json` | 收件箱最近 N 封摘要 |
| `search` | `<query>`（位置参数）, `--json` | 按关键词搜主题/发件人 |
| `read` | `<uid>`（位置参数）, `--json` | 读取指定邮件完整内容 |
| `monitor` | `--interval 300`, `--json` | 轮询新邮件通知，Ctrl+C 退出 |

## 配置推导

```python
imap_host = os.environ.get("EMAIL_IMAP_HOST")
if not imap_host:
    smtp_host = os.environ["EMAIL_SMTP_HOST"]  # e.g. smtp.163.com
    imap_host = smtp_host.replace("smtp.", "imap.", 1)  # → imap.163.com

imap_port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))  # SSL 固定
```

用户名密码复用 `EMAIL_USER` / `EMAIL_PASS`。

## 输出格式

**默认：人类可读文本（stdout）**

- `list`/`search`：序号 + 主题 + 发件人 + 日期 + 正文预览
- `read`：完整头信息 + 正文
- `monitor`：带时间戳的新邮件通知

**`--json`：结构化 JSON（stdout）**

- `list`/`search` → 对象数组
- `read` → 单对象

## MIME 解码

- `email` stdlib 解析 multipart
- 优先 `text/plain`，无则 fallback `text/html`（做简单标签剥离）
- 中文主题/发件人走 `email.header.decode_header`

## 行为细节

- 读取后**自动标记已读**（IMAP SEEN flag）
- 数据库不持久化——每次运行独立连接、独立搜索
- `FROM` 字段走 `email.utils.parseaddr`，取裸地址

## 错误处理

- 连接失败 → 诊断 host/port/网络
- 认证失败 → 提示检查密码/授权码
- 搜索无结果 → 输出 "No matches"（退出码 0）
- Monitor 网络中断 → 重试间隔 60s，不退出

## 测试策略

- 手动验证：`list` → `read <uid>` → `search <keyword>` → `monitor --interval 10`
- 边界：空收件箱、超大邮件、纯 HTML 邮件、多附件邮件
- 错误：无效凭证、断网、无效 UID
