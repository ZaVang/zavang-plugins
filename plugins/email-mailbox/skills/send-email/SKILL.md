---
name: send-email
description: Send an email to the outside world via SMTP. Use when you need to communicate outward — notify someone, send results, alert on completion, or reach a human.
allowed-tools: Bash
---

# Send Email

Your output channel to the outside world. Use this whenever you need to send information out of the Claude Code harness.

## Prerequisites

These 3 environment variables must be set:

| Variable | Description |
|----------|-------------|
| `EMAIL_SMTP_HOST` | SMTP server hostname |
| `EMAIL_USER` | Your email address (also used for SMTP login) |
| `EMAIL_PASS` | SMTP password or authorization code |

Optional (you usually don't need these):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMAIL_SMTP_PORT` | `465` | SMTP port |
| `EMAIL_FROM` | same as `EMAIL_USER` | From address |

## Usage

You choose the recipient each time with `--to=`. There is no default recipient — you decide who to send to.

```
python "<plugin_dir>/scripts/send_email.py" "Subject" "Body" --to=recipient@example.com
```

**HTML email:**
```
python "<plugin_dir>/scripts/send_email.py" "Subject" "<h1>Title</h1><p>content</p>" --html --to=someone@example.com
```

**Body from stdin (for long messages):**
```
echo "Multi-line body..." | python "<plugin_dir>/scripts/send_email.py" "Subject" --to=someone@example.com
```

## Provider Configuration

### NetEase / 网易邮箱 (163.com / 126.com / yeah.net)

```
EMAIL_SMTP_HOST=smtp.163.com    (or smtp.126.com / smtp.yeah.net)
EMAIL_SMTP_PORT=465
EMAIL_USER=your-email@163.com
EMAIL_PASS=<authorization code>
```

How to get the authorization code:
1. Log in to 163/126 webmail
2. Go to Settings → POP3/SMTP/IMAP
3. Enable SMTP service
4. Copy the authorization code shown (it's NOT your login password)

### QQ Mail

```
EMAIL_SMTP_HOST=smtp.qq.com
EMAIL_SMTP_PORT=465
EMAIL_USER=your-email@qq.com
EMAIL_PASS=<authorization code>
```
Enable SMTP in QQ Mail settings → Account → POP3/SMTP service → generate authorization code.

### Gmail

```
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=<app password>
```
Requires 2FA enabled, then generate an App Password at https://myaccount.google.com/apppasswords.

### Outlook / Hotmail

```
EMAIL_SMTP_HOST=smtp-mail.outlook.com
EMAIL_SMTP_PORT=587
EMAIL_USER=your-email@outlook.com
EMAIL_PASS=<your password>
```

## Security

- Never commit credentials. Set env vars in `~/.claude/settings.json` or your shell profile
- Always use authorization codes / app passwords, never your main login password
- The script auto-detects SSL (port 465) vs STARTTLS (port 587), both encrypted
