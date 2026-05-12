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
import re
import json
import time
import imaplib
import email
import argparse
from datetime import datetime
from email.header import decode_header
from email.utils import parseaddr


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
            print("IMAP authentication failed. Check EMAIL_USER and EMAIL_PASS.", file=sys.stderr)
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

    status, _ = server.select(folder, readonly=False)
    if status != "OK":
        print(f"Folder '{folder}' not found or not selectable.", file=sys.stderr)
        sys.exit(1)

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
    query = args.query.replace('"', '')
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
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(json.dumps({
                                "time": now, "uid": uid.decode(),
                                "from": mail_from, "subject": subject
                            }, ensure_ascii=False))
                        else:
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
                except Exception as e:
                    print(f"[Monitor] Reconnect failed: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        if not json_mode:
            print("\n[Monitor] Stopped.")


def main():
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
