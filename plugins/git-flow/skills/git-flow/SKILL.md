---
name: git-flow
description: Install per-repo git discipline for AI agents — an auto-injected issue -> branch -> commit -> PR/MR protocol plus hooks that block editing on main and block agent merges (merging stays human). Works on GitHub (gh) and GitLab (glab), auto-detected. Use when the user wants to enforce a standard git workflow, stop committing directly to main, make agents open PRs/MRs, or set up branch discipline in a repository.
---

# git-flow Setup Skill

Install the git-flow protocol + guard hooks into the **current repository**. After setup,
every Claude Code session in that repo automatically follows the workflow, and two hooks
make the discipline non-skippable. Opt-in is per-repo: only repos where you run this get it.

## What it installs

1. **A SessionStart hook** that injects the protocol into context each session (so the agent
   creates issues / branches / commits / PRs without being told). Forge auto-detected.
2. **A PreToolUse(Edit|Write|NotebookEdit) hook** that blocks edits while on `main`/`master`
   — forcing a feature branch first.
3. **A PreToolUse(Bash) hook** that blocks `gh pr merge` / `glab mr merge` — merging stays
   the human's call.

The *creation* work (issue/branch/commit/PR/MR) is done by the agent following the injected
protocol; the *hooks* are the mechanical guarantees.

## When to Use This Skill

- "set up git-flow / git discipline here", "stop me committing to main"
- "make agents open PRs/MRs instead of pushing to main"
- "enforce branch + review workflow in this repo"

## Workflow

### Step 1: Confirm repo + detect forge + check auth

1. Confirm the cwd is inside a git repo: `git rev-parse --show-toplevel`. If not, offer `git init` or stop.
2. Detect the forge from the origin remote: `git remote get-url origin`
   - contains `github` → **GitHub**: the CLI is `gh`, review unit is **PR**
   - contains `gitlab` → **GitLab**: the CLI is `glab`, review unit is **MR**
   - no/unknown remote → **local-only**: skip issue/PR; the protocol degrades to branch + human-merge
3. If a forge was detected, check auth and tell the user if they must log in:
   - GitHub: `gh auth status`
   - GitLab: `glab auth status`

### Step 2: Install the hook scripts

Create `<repo>/.claude/hooks/` and copy the three scripts from this skill's
`templates/hooks/` directory into it (verbatim):

```
<repo>/.claude/hooks/
├── gitflow_session.sh
├── gitflow_branch_guard.sh
└── gitflow_merge_guard.sh
```

### Step 3: Register the hooks in project settings

Merge the `hooks` block from this skill's `templates/settings.snippet.json` into
`<repo>/.claude/settings.json` (create the file if absent). **Merge, do not overwrite** —
if `settings.json` already has a `hooks` key, append these entries to the existing
`SessionStart` / `PreToolUse` arrays rather than replacing them.

The registered commands are relative to the repo root:
- `bash .claude/hooks/gitflow_session.sh`
- `bash .claude/hooks/gitflow_branch_guard.sh`
- `bash .claude/hooks/gitflow_merge_guard.sh`

### Step 4: Tell the user it's active

Report:
- Forge detected and whether `gh`/`glab` is authenticated.
- That the hooks fire automatically from the next session/tool-call in this repo.
- The **escape hatch**: to temporarily allow direct edits on main and agent merges in this
  repo, create an empty file `.claude/gitflow.allow-main` (delete it to re-arm).
- A one-line smoke check they can run now to confirm the branch guard works:
  `git rev-parse --abbrev-ref HEAD` (if it prints main/master, an Edit will be blocked).

## The protocol the agent will follow (after setup)

For any code-change task:
1. **Issue** — read the issue the user gave you; else create one **only if non-trivial**
   (skip typos/one-liners). `gh issue create` / `glab issue create`.
2. **Branch** — never edit on main/master. `git checkout -b <type>/<short-desc>`.
3. **Implement**, then **commit** with a clear Conventional-Commits message (never skip it).
4. **PR/MR** — `gh pr create` / `glab mr create`, real description, link issue via "Closes #<n>".
5. **Stop before merge** — the human merges.

## Important Rules for AI

1. **Never merge** on the user's behalf in a git-flow repo — open the PR/MR and hand back.
2. **Pick the CLI by forge**: `gh` for GitHub, `glab` for GitLab. Don't hardcode one.
3. **Flag differences**: GitHub `gh ... --body` vs GitLab `glab ... --description`; PR `--base`
   vs MR `--target-branch`. The closing keyword "Closes #<n>" works on both.
4. **Local-only repos**: no remote → skip issue/PR/MR; still branch and let the human merge.
5. When merging settings, preserve any existing hooks the project already had.
