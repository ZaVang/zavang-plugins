#!/usr/bin/env bash
# git-flow :: PreToolUse hook for Edit|Write|NotebookEdit
# Blocks edits while on main/master, forcing a feature branch first.
# Escape hatch: create .claude/gitflow.allow-main in the repo to bypass.
# exit 2 => block the tool call and surface this message to the agent.

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
[ -f "$root/.claude/gitflow.allow-main" ] && exit 0

branch=$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
case "$branch" in
  main|master)
    echo "[git-flow] Blocked: you are on '$branch'. Create a feature branch first (git checkout -b <type>/<short-desc>) and retry. If the user explicitly wants to edit '$branch' directly, create the file .claude/gitflow.allow-main to bypass for this repo." >&2
    exit 2
    ;;
esac
exit 0
