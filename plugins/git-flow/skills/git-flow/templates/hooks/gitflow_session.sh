#!/usr/bin/env bash
# git-flow :: SessionStart hook
# Injects the git-flow protocol into the agent's context at session start.
# Forge (GitHub/GitLab) is auto-detected from the origin remote.
# Installed per-repo into .claude/hooks/ by the git-flow setup skill.

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0

remote=$(git -C "$root" remote get-url origin 2>/dev/null || true)
case "$remote" in
  *github*) forge="GitHub — use the 'gh' CLI; the review unit is a PR";;
  *gitlab*) forge="GitLab — use the 'glab' CLI; the review unit is an MR";;
  *)        forge="no recognized remote — skip issue/PR, just branch + let the human merge locally";;
esac

cat <<EOF
[git-flow] active in this repo. For ANY code-change task, follow this protocol:
- Forge: ${forge}
- Issue: if the user pointed you at an issue, read it. If none AND the task is non-trivial, create one. Skip for trivial fixes (typos / one-liners).
- Branch: NEVER edit on main/master (a hook blocks it). Create a branch first: <type>/<short-desc> (e.g. fix/login-null, feat/export-csv).
- Commit: write a clear Conventional-Commits message yourself — never skip it.
- PR/MR: when done, open one with a real description; link the issue with "Closes #<n>".
- Merge: DO NOT merge. Open the PR/MR and stop — the human merges (a hook blocks agent merges).
EOF
exit 0
