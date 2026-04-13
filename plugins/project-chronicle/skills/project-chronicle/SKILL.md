---
name: project-chronicle
description: |
  Records a summary of the current Claude Code session into a project development log (docs/chronicle.md).
  Use this skill when: (1) user runs /chronicle, (2) user says "record this session", "update project log",
  "save what we did", "记录这次session", "更新项目日志", (3) user wraps up a significant work session and
  wants to document progress. Appends a new timestamped entry each time, building a cumulative project
  history across multiple sessions. Supports two modes: concise (default, ~10-15 lines per session) and
  detailed (includes decisions, problems, and solutions). Never records every conversation turn—only
  meaningful phases and outcomes.
author: Claude Code
version: 1.0.0
date: 2026-03-16
---

# Project Chronicle

Records how you built a project with Claude Code — capturing the development journey across sessions so you can look back and understand the path you took.

## When to use

Invoke at the end of a work session when something meaningful was accomplished. Each invocation appends one entry to `docs/chronicle.md`. Over time, these entries build a full project narrative.

## How to invoke

```
/chronicle          → concise summary (default)
/chronicle detail   → detailed summary with decisions and problems
```

## What to record

Scan back through the conversation to identify:

**Always include (concise mode):**
- Session date and a one-line objective
- Major phases (3-6 bullet points, not individual messages)
- Skills, MCPs, or plugins that were actually invoked
- Final outcome/status

**Add in detail mode:**
- Key architectural or design decisions made, and why
- Significant problems encountered and how they were resolved
- Approaches tried that didn't work (valuable for future reference)
- Any skills or knowledge that might be worth extracting with `/claudeception`

**What to skip:**
- Back-and-forth debug iterations (summarize as "debugged X, resolved by Y")
- Failed attempts unless they led to important insight
- Routine tool calls that were just execution steps

## Output format

Append to `docs/chronicle.md` in the current project directory. Create the file and `docs/` folder if they don't exist.

Use this structure:

```markdown
## Session [N] — YYYY-MM-DD

**Objective**: One sentence describing what this session set out to do.

**Steps**:
1. [Phase or milestone, not individual messages]
2. ...

**Tools & Skills used**: [List any skills, MCPs, or notable tools invoked]

**Outcome**: What was accomplished; current status of the project.
```

For detail mode, add:

```markdown
**Key decisions**:
- [Decision]: [Brief rationale]

**Problems & solutions**:
- [Problem encountered]: [How it was resolved]
```

### Session numbering

Read `docs/chronicle.md` to find the last session number and increment. If the file doesn't exist, start at Session 1.

### Header on first session

When creating the file for the first time, add a project header before the first session entry:

```markdown
# Project Chronicle: [project name — infer from directory name or package.json]

> Auto-generated development log. Each entry summarizes one Claude Code session.

---
```

## Example entry (concise)

```markdown
## Session 3 — 2026-03-16

**Objective**: Add authentication flow and user profile API.

**Steps**:
1. Designed JWT-based auth schema and discussed token expiry strategy
2. Implemented login/logout endpoints with middleware
3. Added user profile CRUD endpoints
4. Wrote integration tests and fixed edge cases with token refresh

**Tools & Skills used**: `superpowers:tdd`, `llm-bridge` for user intent parsing

**Outcome**: Auth system complete and tested. Profile API working. Ready to start frontend integration.
```

## Example entry (detail mode)

```markdown
## Session 2 — 2026-03-15

**Objective**: Set up project infrastructure and CI pipeline.

**Steps**:
1. Initialized monorepo with Turborepo
2. Configured ESLint, TypeScript, and shared config packages
3. Set up GitHub Actions for lint + test on PR
4. Debugged failing CI runs

**Tools & Skills used**: None explicitly

**Outcome**: CI pipeline green. Monorepo structure established.

**Key decisions**:
- Turborepo over Nx: simpler config, sufficient for project scale
- Shared tsconfig in packages/config to avoid duplication

**Problems & solutions**:
- CI failing due to Node version mismatch: pinned to Node 20 in workflow YAML
- ESLint config not resolving workspace packages: added `moduleNameMapper` to tsconfig paths
```

## Notes

- If there are multiple recent sessions you can tell apart from the conversation (e.g., conversation was clearly split into unrelated chunks), you may create multiple entries in one invocation.
- This skill is about your *process*, not the code itself — think of it as a captain's log, not a commit message.
- For extracting reusable patterns discovered during the session into skills, use `/claudeception` afterward.
