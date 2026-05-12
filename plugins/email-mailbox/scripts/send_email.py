#!/usr/bin/env python3
"""Send email via SMTP. Reads config from environment variables.

Required env vars:
  EMAIL_SMTP_HOST  - SMTP server hostname
  EMAIL_USER       - SMTP login username / email
  EMAIL_PASS       - SMTP password or authorization code

Optional env vars:
  EMAIL_SMTP_PORT  - SMTP port (default: 465)
  EMAIL_SMTP_MODE  - "ssl" (port 465) or "starttls" (port 587). Auto-detected from port if unset.
  EMAIL_FROM       - From address (default: same as EMAIL_USER)
  EMAIL_TO         - Default recipient (can be overridden with --to=)

Usage:
  python send_email.py "Subject" "Body" --to=someone@example.com
  echo "Body from stdin" | python send_email.py "Subject" --to=someone@example.com
  python send_email.py "Subject" --html "<h1>HTML</h1>" --to=someone@example.com
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def main():
    args = sys.argv[1:]

    to_addr = None
    is_html = False
    positional = []
    for a in args:
        if a == "--html":
            is_html = True
        elif a.startswith("--to="):
            to_addr = a[5:]
        elif a.startswith("--cc="):
            pass
        else:
            positional.append(a)

    if not positional:
        print("Usage: send_email.py <subject> [body] --to=<addr> [--html]", file=sys.stderr)
        sys.exit(1)

    subject = positional[0]

    if len(positional) >= 2:
        body = positional[1]
    else:
        body = sys.stdin.read()

    smtp_host = os.environ.get("EMAIL_SMTP_HOST")
    smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "465"))
    smtp_mode = os.environ.get("EMAIL_SMTP_MODE")  # "ssl" or "starttls", auto if unset
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    from_addr = os.environ.get("EMAIL_FROM", user)
    to_addr = to_addr or os.environ.get("EMAIL_TO")

    missing = []
    if not smtp_host:
        missing.append("EMAIL_SMTP_HOST")
    if not user:
        missing.append("EMAIL_USER")
    if not password:
        missing.append("EMAIL_PASS")
    if not to_addr:
        missing.append("recipient (use --to= or set EMAIL_TO)")
    if missing:
        print(f"Missing config: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect SSL vs STARTTLS from port if mode not explicitly set
    if smtp_mode is None:
        smtp_mode = "ssl" if smtp_port == 465 else "starttls"

    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject

    subtype = "html" if is_html else "plain"
    msg.attach(MIMEText(body, subtype, "utf-8"))

    try:
        if smtp_mode == "ssl":
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
            server.starttls()

        server.login(user, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()
        print(f"Email sent to {to_addr} (via {smtp_host}:{smtp_port}/{smtp_mode})")
    except smtplib.SMTPAuthenticationError:
        print("SMTP authentication failed. Check EMAIL_USER and EMAIL_PASS.", file=sys.stderr)
        sys.exit(1)
    except smtplib.SMTPConnectError:
        print(f"Cannot connect to {smtp_host}:{smtp_port}. Check EMAIL_SMTP_HOST, port, and network.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to send: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
