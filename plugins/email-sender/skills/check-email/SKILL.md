---
name: check-email
description: Check and read emails via IMAP. Use when you need to read inbox, search for specific emails, or monitor for new messages. Supports 163.com, QQ Mail, Gmail, Outlook.
allowed-tools: Bash
---

# Check Email

Your input channel from the outside world. Use this whenever you need to read emails, check for replies, or monitor incoming messages.

## Prerequisites

Uses the same credentials as send-email. These 3 environment variables must be set:

| Variable | Description |
|----------|-------------|
| `EMAIL_SMTP_HOST` | SMTP server hostname (IMAP host auto-derived: smtp. → imap.) |
| `EMAIL_USER` | Your email address (also used for IMAP login) |
| `EMAIL_PASS` | SMTP/IMAP password or authorization code |

Optional:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMAIL_IMAP_HOST` | derived from `EMAIL_SMTP_HOST` | Explicit IMAP server override |
| `EMAIL_IMAP_PORT` | `993` | IMAP port |
| `EMAIL_IMAP_PASS` | `EMAIL_PASS` | IMAP password override (for providers with separate IMAP/SMTP credentials, e.g. 163.com) |

## Usage

All commands use the same script with subcommands:

```bash
python "<plugin_dir>/scripts/check_email.py" <subcommand> [options]
```

### List recent emails

```bash
python "<plugin_dir>/scripts/check_email.py" list --limit 10
python "<plugin_dir>/scripts/check_email.py" list --limit 5 --json
python "<plugin_dir>/scripts/check_email.py" list --folder INBOX
```

Output (text mode):
```
[1] Re: Project Update | alice@example.com | 2026-05-12 14:30
    Preview of the email body first line...
    Preview of the email body second line...

[2] Meeting Notes | bob@corp.com | 2026-05-12 10:15
    Here are the notes from today's meeting...
```

Output (--json mode):
```json
[
  {"uid": "123", "from": "alice@example.com", "subject": "Re: Project Update", "date": "2026-05-12 14:30", "preview": "Preview of the email body..."},
  {"uid": "124", "from": "bob@corp.com", "subject": "Meeting Notes", "date": "2026-05-12 10:15", "preview": "Here are the notes..."}
]
```

### Search emails

```bash
python "<plugin_dir>/scripts/check_email.py" search "keyword"
python "<plugin_dir>/scripts/check_email.py" search "alice" --json
```

Searches subject and sender fields using IMAP SEARCH.

### Read a specific email

```bash
python "<plugin_dir>/scripts/check_email.py" read 123
python "<plugin_dir>/scripts/check_email.py" read 123 --json
```

Output (text mode):
```
From: alice@example.com
Date: 2026-05-12 14:30
Subject: Re: Project Update
---
Full email body text here...
```

Output (--json mode):
```json
{"uid": "123", "from": "alice@example.com", "subject": "Re: Project Update", "date": "2026-05-12 14:30", "body": "Full email body text here..."}
```

### Monitor for new emails

```bash
python "<plugin_dir>/scripts/check_email.py" monitor --interval 300
python "<plugin_dir>/scripts/check_email.py" monitor --interval 60 --json
```

Polls inbox every N seconds (default 300). Prints notification when new mail arrives. Press Ctrl+C to stop.

## Provider IMAP Servers

| Provider | SMTP (set as EMAIL_SMTP_HOST) | IMAP (auto-derived) |
|----------|------------------------------|---------------------|
| 163.com | smtp.163.com | imap.163.com |
| 126.com | smtp.126.com | imap.126.com |
| QQ Mail | smtp.qq.com | imap.qq.com |
| Gmail | smtp.gmail.com | imap.gmail.com |
| Outlook | smtp-mail.outlook.com | imap-mail.outlook.com |

Note: For Outlook, the auto-derivation (smtp-mail. → imap-mail.) works — no special handling needed.

## Security

- Same credentials as send-email — no extra env vars needed
- IMAP SSL (port 993) enforced
- Emails read are marked as SEEN on the server
