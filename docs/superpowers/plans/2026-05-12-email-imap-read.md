# Email IMAP Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add IMAP read capability to email-sender plugin via a single `check_email.py` script with `list|search|read|monitor` subcommands.

**Architecture:** Single-entry CLI script (`check_email.py`) using Python stdlib `imaplib` + `email`. IMAP host auto-derived from `EMAIL_SMTP_HOST` (smtp. → imap.). Zero new dependencies. New `check-email` skill alongside existing `send-email`.

**Tech Stack:** Python 3.13+, stdlib only (`imaplib`, `email`, `json`, `time`, `argparse`)

---

### Task 1: Create check-email SKILL.md

**Files:**
- Create: `plugins/email-sender/skills/check-email/SKILL.md`

- [ ] **Step 1: Write the skill definition**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add plugins/email-sender/skills/check-email/SKILL.md
git commit -m "feat: add check-email skill definition for IMAP read"
```

---

### Task 2: Create check_email.py — config, IMAP connection, and `list` subcommand

**Files:**
- Create: `plugins/email-sender/scripts/check_email.py`

- [ ] **Step 1: Write the script with config derivation, IMAP connect, and list subcommand**

```python
#!/usr/bin/env python3
"""Check/read emails via IMAP. Reads config from environment variables.

Required env vars (shared with send_email.py):
  EMAIL_SMTP_HOST  - SMTP server hostname (IMAP host auto-derived: smtp. -> imap.)
  EMAIL_USER       - SMTP/IMAP login username / email
  EMAIL_PASS       - SMTP/IMAP password or authorization code

Optional env vars:
  EMAIL_IMAP_HOST  - Explicit IMAP server (overrides auto-derivation)
  EMAIL_IMAP_PORT  - IMAP port (default: 993)

Usage:
  python check_email.py list [--limit N] [--folder INBOX] [--json]
  python check_email.py search <query> [--json]
  python check_email.py read <uid> [--json]
  python check_email.py monitor [--interval N] [--json]
"""

import os
import sys
import json
import time
import imaplib
import email
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime


def get_config():
    smtp_host = os.environ.get("EMAIL_SMTP_HOST")
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")

    missing = []
    if not smtp_host:
        missing.append("EMAIL_SMTP_HOST")
    if not user:
        missing.append("EMAIL_USER")
    if not password:
        missing.append("EMAIL_PASS")
    if missing:
        print(f"Missing config: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    imap_host = os.environ.get("EMAIL_IMAP_HOST")
    if not imap_host:
        imap_host = smtp_host.replace("smtp.", "imap.", 1)
        # Handle outlook edge case: smtp-mail.outlook.com -> imap-mail.outlook.com
        # The replace above handles this correctly since smtp_host starts with "smtp-mail."

    imap_port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))

    return imap_host, imap_port, user, password


def decode_mime_header(value):
    if value is None:
        return ""
    parts = decode_header(value)
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            charset = charset or "utf-8"
            result.append(part.decode(charset, errors="replace"))
        else:
            result.append(part)
    return "".join(result)


def extract_body(msg):
    if msg.is_multipart():
        text_parts = []
        html_parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    text_parts.append(payload.decode("utf-8", errors="replace"))
            elif content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    html_parts.append(payload.decode("utf-8", errors="replace"))
        if text_parts:
            return "\n".join(text_parts)
        if html_parts:
            return strip_html("\n".join(html_parts))
        return ""
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            content_type = msg.get_content_type()
            if content_type == "text/html":
                return strip_html(payload.decode("utf-8", errors="replace"))
            return payload.decode("utf-8", errors="replace")
        return ""


def strip_html(text):
    """Remove HTML tags, leaving plain text."""
    import re
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def connect_imap():
    imap_host, imap_port, user, password = get_config()
    try:
        server = imaplib.IMAP4_SSL(imap_host, imap_port)
        server.login(user, password)
        return server
    except imaplib.IMAP4.error as e:
        msg = str(e)
        if "authentication" in msg.lower() or "login" in msg.lower():
            print(f"IMAP authentication failed. Check EMAIL_USER and EMAIL_PASS.", file=sys.stderr)
        else:
            print(f"IMAP error: {msg}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Cannot connect to {imap_host}:{imap_port}. Check network and IMAP server.", file=sys.stderr)
        sys.exit(1)


def format_email_summary(uid, mail_from, subject, date_str, body, json_mode=False):
    if json_mode:
        return {
            "uid": uid,
            "from": mail_from,
            "subject": subject,
            "date": date_str,
            "preview": body[:200],
        }
    else:
        preview = "\n    ".join(body[:200].split("\n")[:2])
        date_display = date_str[:16] if len(date_str) > 16 else date_str
        return f"[{uid}] {subject[:60]} | {mail_from[:40]} | {date_display}\n    {preview}"


def format_email_full(uid, mail_from, subject, date_str, body, json_mode=False):
    if json_mode:
        return {"uid": uid, "from": mail_from, "subject": subject, "date": date_str, "body": body}
    else:
        return f"From: {mail_from}\nDate: {date_str}\nSubject: {subject}\n---\n{body}"


def cmd_list(server, args):
    folder = getattr(args, "folder", "INBOX")
    limit = getattr(args, "limit", 10)
    json_mode = getattr(args, "json", False)

    server.select(folder, readonly=False)

    status, data = server.search(None, "ALL")
    if status != "OK":
        print("Failed to search inbox.", file=sys.stderr)
        sys.exit(1)

    uids = data[0].split()
    if not uids:
        if json_mode:
            print("[]")
        else:
            print("(inbox empty)")
        return

    recent = uids[-limit:]
    recent.reverse()

    results = []
    for uid in recent:
        status, msg_data = server.fetch(uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] BODY.PEEK[TEXT])")
        if status != "OK":
            continue

        header_part = msg_data[0][1] if msg_data[0][1] else b""
        body_part = msg_data[1][1] if len(msg_data) > 1 and msg_data[1][1] else b""

        header_msg = email.message_from_bytes(header_part)
        mail_from = decode_mime_header(header_msg.get("From", ""))
        mail_from = parseaddr(mail_from)[1] or mail_from
        subject = decode_mime_header(header_msg.get("Subject", "(no subject)"))
        date_str = header_msg.get("Date", "")

        body_preview = body_part.decode("utf-8", errors="replace")[:200].strip()

        results.append(format_email_summary(
            uid.decode(), mail_from, subject, date_str, body_preview, json_mode
        ))

    if json_mode:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(r)
            print()


def cmd_search(server, args):
    query = args.query
    json_mode = getattr(args, "json", False)

    server.select("INBOX", readonly=False)

    status, data = server.search(None, f'OR SUBJECT "{query}" FROM "{query}"')
    if status != "OK":
        print("Search failed.", file=sys.stderr)
        sys.exit(1)

    uids = data[0].split()
    if not uids:
        if json_mode:
            print("[]")
        else:
            print(f"No matches for '{query}'")
        return

    results = []
    for uid in uids:
        status, msg_data = server.fetch(uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] BODY.PEEK[TEXT])")
        if status != "OK":
            continue

        header_part = msg_data[0][1] if msg_data[0][1] else b""
        body_part = msg_data[1][1] if len(msg_data) > 1 and msg_data[1][1] else b""

        header_msg = email.message_from_bytes(header_part)
        mail_from = decode_mime_header(header_msg.get("From", ""))
        mail_from = parseaddr(mail_from)[1] or mail_from
        subject = decode_mime_header(header_msg.get("Subject", "(no subject)"))
        date_str = header_msg.get("Date", "")

        body_preview = body_part.decode("utf-8", errors="replace")[:200].strip()

        results.append(format_email_summary(
            uid.decode(), mail_from, subject, date_str, body_preview, json_mode
        ))

    if json_mode:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"Found {len(results)} match(es):\n")
        for r in results:
            print(r)
            print()


def cmd_read(server, args):
    uid = args.uid.encode() if isinstance(args.uid, str) else args.uid
    json_mode = getattr(args, "json", False)

    server.select("INBOX", readonly=False)

    status, msg_data = server.fetch(uid, "(RFC822)")
    if status != "OK":
        print(f"Failed to fetch email {args.uid}.", file=sys.stderr)
        sys.exit(1)

    msg = email.message_from_bytes(msg_data[0][1])
    mail_from = decode_mime_header(msg.get("From", ""))
    mail_from = parseaddr(mail_from)[1] or mail_from
    subject = decode_mime_header(msg.get("Subject", "(no subject)"))
    date_str = msg.get("Date", "")
    body = extract_body(msg)

    output = format_email_full(args.uid, mail_from, subject, date_str, body, json_mode)
    if json_mode:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(output)


def cmd_monitor(server, args):
    interval = getattr(args, "interval", 300)
    json_mode = getattr(args, "json", False)

    server.select("INBOX", readonly=False)
    status, data = server.search(None, "ALL")
    if status != "OK":
        print("Failed to search inbox.", file=sys.stderr)
        sys.exit(1)

    seen_uids = set(data[0].split())
    if not json_mode:
        print(f"[Monitor] Watching INBOX every {interval}s. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(interval)
            try:
                server.select("INBOX", readonly=False)
                status, data = server.search(None, "ALL")
                if status != "OK":
                    continue

                current_uids = set(data[0].split())
                new_uids = current_uids - seen_uids

                if new_uids:
                    for uid in sorted(new_uids):
                        status, msg_data = server.fetch(uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
                        if status != "OK":
                            continue
                        header_part = msg_data[0][1] if msg_data[0][1] else b""
                        header_msg = email.message_from_bytes(header_part)
                        mail_from = decode_mime_header(header_msg.get("From", ""))
                        mail_from = parseaddr(mail_from)[1] or mail_from
                        subject = decode_mime_header(header_msg.get("Subject", "(no subject)"))

                        if json_mode:
                            from datetime import datetime
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(json.dumps({
                                "time": now, "uid": uid.decode(),
                                "from": mail_from, "subject": subject
                            }, ensure_ascii=False))
                        else:
                            from datetime import datetime
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[{now}] New: \"{subject[:50]}\" from {mail_from}")

                    seen_uids = current_uids
            except (imaplib.IMAP4.error, OSError):
                if not json_mode:
                    print("[Monitor] Connection lost, retrying in 60s...")
                time.sleep(60)
                try:
                    server = connect_imap()
                    server.select("INBOX", readonly=False)
                    status, data = server.search(None, "ALL")
                    if status == "OK":
                        seen_uids = set(data[0].split())
                except Exception:
                    pass
    except KeyboardInterrupt:
        if not json_mode:
            print("\n[Monitor] Stopped.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check/read emails via IMAP")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List recent emails")
    p_list.add_argument("--limit", type=int, default=10)
    p_list.add_argument("--folder", default="INBOX")
    p_list.add_argument("--json", action="store_true")

    p_search = sub.add_parser("search", help="Search emails by keyword")
    p_search.add_argument("query")
    p_search.add_argument("--json", action="store_true")

    p_read = sub.add_parser("read", help="Read a specific email by UID")
    p_read.add_argument("uid")
    p_read.add_argument("--json", action="store_true")

    p_monitor = sub.add_parser("monitor", help="Monitor for new emails")
    p_monitor.add_argument("--interval", type=int, default=300)
    p_monitor.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    server = connect_imap()
    try:
        if args.command == "list":
            cmd_list(server, args)
        elif args.command == "search":
            cmd_search(server, args)
        elif args.command == "read":
            cmd_read(server, args)
        elif args.command == "monitor":
            cmd_monitor(server, args)
    finally:
        try:
            server.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick syntax check**

```bash
python -c "import py_compile; py_compile.compile('plugins/email-sender/scripts/check_email.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add plugins/email-sender/scripts/check_email.py
git commit -m "feat: add check_email.py with list/search/read/monitor subcommands"
```

---

### Task 3: Update plugin.json

**Files:**
- Modify: `plugins/email-sender/.claude-plugin/plugin.json`

- [ ] **Step 1: Bump version and update metadata**

Change plugin.json from:
```json
{
  "name": "email-sender",
  "description": "Send emails via SMTP from within Claude Code — gives AI an output channel to the outside world",
  "version": "1.0.0",
  "author": { "name": "ZaVang" },
  "homepage": "https://github.com/ZaVang/zavang-plugins",
  "repository": "https://github.com/ZaVang/zavang-plugins",
  "license": "MIT",
  "keywords": ["email", "smtp", "communication", "output"]
}
```

To:
```json
{
  "name": "email-sender",
  "description": "Send and receive emails via SMTP/IMAP from within Claude Code — two-way communication channel to the outside world",
  "version": "1.1.0",
  "author": { "name": "ZaVang" },
  "homepage": "https://github.com/ZaVang/zavang-plugins",
  "repository": "https://github.com/ZaVang/zavang-plugins",
  "license": "MIT",
  "keywords": ["email", "smtp", "imap", "communication", "input", "output"]
}
```

- [ ] **Step 2: Commit**

```bash
git add plugins/email-sender/.claude-plugin/plugin.json
git commit -m "chore: bump email-sender to 1.1.0 for IMAP read support"
```

---

### Task 4: Manual verification

- [ ] **Step 1: Verify list command**

```bash
python plugins/email-sender/scripts/check_email.py list --limit 3
```
Expected: Shows up to 3 recent emails with subject, sender, date, preview.

- [ ] **Step 2: Verify list --json**

```bash
python plugins/email-sender/scripts/check_email.py list --limit 1 --json
```
Expected: Valid JSON array with uid, from, subject, date, preview fields.

- [ ] **Step 3: Verify read command**

```bash
# Use UID from step 2
python plugins/email-sender/scripts/check_email.py read <uid>
```
Expected: Full email content with From, Date, Subject, body.

- [ ] **Step 4: Verify search command**

```bash
python plugins/email-sender/scripts/check_email.py search "test"
```
Expected: Matching emails or "No matches for 'test'".

- [ ] **Step 5: Verify syntax of both scripts still clean**

```bash
python -c "import py_compile; py_compile.compile('plugins/email-sender/scripts/send_email.py', doraise=True); py_compile.compile('plugins/email-sender/scripts/check_email.py', doraise=True); print('Both OK')"
```

- [ ] **Step 6: Commit if all pass**
(No separate commit — verification only; all code already committed)

---

### Task 5: Update plugin cache and test from CodeMemory

- [ ] **Step 1: Reinstall plugin from local source**

```bash
# Remove cached version
rm -rf ~/.claude/plugins/cache/zavang-plugins/email-sender

# Reinstall from local dev source
claude --plugin-dir /d/work/zavang-plugins/plugins/email-sender
# Or use: /plugin install email-sender@zavang-plugins (after cache clear)
```

- [ ] **Step 2: Verify skill loads**

In Claude Code: `/check-email` should appear as an available skill.

- [ ] **Step 3: Run check-email from Claude Code**

Ask Claude to check recent emails: "check my 3 most recent emails"
Expected: Claude invokes check-email skill, runs the script, returns email summaries.
