#!/usr/bin/env bash
# git-flow :: PreToolUse hook for Bash
# Blocks agent-initiated merges (gh pr merge / glab mr merge) — merging is the
# human's call. Reads the hook event JSON from stdin and greps the command.
# Escape hatch: create .claude/gitflow.allow-main in the repo to bypass.

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
[ -f "$root/.claude/gitflow.allow-main" ] && exit 0

payload=$(cat 2>/dev/null)
if printf '%s' "$payload" | grep -Eq 'gh +pr +merge|glab +mr +merge'; then
  echo "[git-flow] Blocked: merging is the human's job in this repo. Open the PR/MR and stop — let the user merge. (Bypass for this repo: create .claude/gitflow.allow-main)" >&2
  exit 2
fi
exit 0
