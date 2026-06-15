# git-flow — Per-Repo Git Discipline for AI Agents

Stops the "edit on main, skip the commit message" habit, and prepares a repo for
multi-agent work — without heavy CI ceremony. Opt-in per repository.

## The four guarantees it delivers

1. **main stays clean** — editing on main/master is blocked; work happens on a branch.
2. **traceable to intent** — each task reads or creates an issue; PR/MR links it (`Closes #n`).
3. **isolated until reviewed** — work lives on a branch (or worktree) until merged.
4. **nothing merges unreviewed** — the agent opens the PR/MR and stops; **the human merges**.

The exact ceremony is light: for trivial fixes the issue step is skipped; there is no required
CI — the "verification gate" is whatever the agent can run locally (tests/lint/build if they
exist, otherwise an AI review of the diff and a run of the app).

## How it works: agent does, hooks guarantee

| Step | Who | Mechanism |
|------|-----|-----------|
| read/create issue | **agent** | injected protocol → `gh`/`glab` |
| create branch | **agent** | injected protocol (+ hook backstop) |
| implement + commit (good message) | **agent** | injected protocol |
| open PR/MR | **agent** | injected protocol → `gh`/`glab` |
| **merge** | **human** | agent merges are hook-blocked |
| don't edit on main | — | PreToolUse hook blocks it |

The SessionStart hook injects the protocol every session (so it's automatic, no command to
remember). The two PreToolUse hooks are the mechanical backstops.

## GitHub vs GitLab — one plugin, auto-detected

Detected from `git remote get-url origin`:

| | GitHub (`gh`) | GitLab (`glab`) |
|---|---|---|
| review unit | PR | MR |
| create issue | `gh issue create --title --body` | `glab issue create --title --description` |
| open review | `gh pr create --base main` | `glab mr create --target-branch main` |
| close on merge | `Closes #n` in body | `Closes #n` in description |

The guard hooks are forge-agnostic (pure git + a grep for both `gh pr merge` and
`glab mr merge`). Local-only repos (no remote) degrade gracefully to branch + human-merge.

## Escape hatch

To allow direct edits on main and agent merges in a repo (e.g. a deliberate one-off), create
an empty file `.claude/gitflow.allow-main`. Delete it to re-arm the guards.

## Install

Run the `git-flow` skill in the target repo ("set up git-flow here"). It detects the forge,
checks `gh`/`glab` auth, copies the hook scripts into `.claude/hooks/`, and registers them in
`.claude/settings.json`. From the next session the discipline is automatic in that repo only.

## Composes with the loops

git-flow is the outer shell around multi-ralph / product-loop: those run on a branch, their
Evaluator is the local verification gate, completion opens a PR/MR, and you merge. The guard
hooks keep even the loops on-protocol.
