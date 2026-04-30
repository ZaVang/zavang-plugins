# Product-Loop Evolution Agent — Design

**Date**: 2026-04-30
**Status**: Design

## Overview

Add a `--mode` option to product-loop, letting the user choose which reviewer audits in Step A:

| Mode | Reviewer in Step A |
|------|--------------------|
| `experience` (default) | Product Experience Reviewer (current behavior) |
| `evolution` | Product Evolution Reviewer (new) |
| `all` | Both run in parallel |

## New System Agent: `product-evolution-reviewer`

New file: `plugins/product-loop/agents/product-evolution-reviewer/AGENT.md`

**Identity**: A product strategist and feature ideator. Takes a demo/prototype and figures out what features would turn it into a compelling, complete product. Not a code auditor — a product thinker.

**Four primary dimensions** (equal weight):

1. **Core Completeness** — What's missing to make the core loop feel like a real, shippable product? (onboarding, settings, error recovery, empty states, data persistence, undo, bulk actions, search, filters)

2. **Competitive Gap** — What table-stakes features do similar products have that this doesn't? Research competitors via WebFetch/WebSearch.

3. **Feature Deepening** — Where can existing features go deeper? Power-user shortcuts, customization, integrations, advanced workflows, social/collaboration features.

4. **Differentiation** — What unique/"wow" features would set this apart? "If only it could..." proposals competitors can't easily copy.

**Secondary section**: Technical Health — brief scan of architecture risks, performance bottlenecks, test gaps — framed as "things that matter as features grow," not a full code audit.

**Output**: `docs/orch/evolution-audit-report.md`

**Tools**: Read, Write, Bash, Glob, Grep, WebFetch, WebSearch (same as experience reviewer — needs web for competitor research)

**Report format** mirrors the experience reviewer but with evolution-focused phases:

```
Executive Summary — overall product maturity score /10

Phase 1: Core Completeness — what's missing from the core loop
Phase 2: Competitive Gap — what competitors have that we don't
Phase 3: Feature Deepening — where existing features can go deeper
Phase 4: Differentiation & Wow Factor — unique feature proposals

Technical Health (secondary) — architecture/performance risks

Prioritized Recommendations:
  🔴 Critical — missing table-stakes features
  🟡 Important — features that significantly improve completeness
  🟢 Nice-to-have — power-user features, polish
  💡 Feature Idea — differentiation/wow-factor proposals
```

## Modified Files

### `skills/product-loop/SKILL.md`

**1. Usage section** — add `--mode` parameter:

```bash
/product-loop:product-loop
/product-loop:product-loop --mode evolution
/product-loop:product-loop --mode all
/product-loop:product-loop --mode experience --max-iter 3
```

**2. Parameter parsing** — add alongside `--max-iter` and `--sprint`:

```
--mode MODE — reviewer mode: experience | evolution | all (default: experience)
```

**3. Shared knowledge files table** — add `evolution-audit-report.md` row:

| File | Writer | Reader | Purpose |
|------|--------|--------|---------|
| `docs/orch/evolution-audit-report.md` | Evolution Reviewer (Step A) | Planner (Step B) | Evolution audit report |

**4. Step A** — branch on mode:

```
Mode "experience":  launch agent product-experience-reviewer → product-audit-report.md
Mode "evolution":   launch agent product-evolution-reviewer  → evolution-audit-report.md
Mode "all":         launch BOTH in parallel → both reports
```

For `all` mode, both Agent calls are made in the same message (no dependencies between them).

**5. Step B Planner prompt** — conditional input section:

- `experience`: reads `product-audit-report.md` (unchanged)
- `evolution`: reads `evolution-audit-report.md` instead
- `all`: reads both reports, synthesizes a unified plan, writes negotiation responses to both reviewers in `negotiation.md`

**6. Post-loop summary** — mode-aware: references the correct report(s)

### `commands/help.md`

- Add `--mode` to usage examples
- Add prerequisite: `product-evolution-reviewer` agent needed for `evolution` / `all` modes
- Add Evolution Agent to the comparison table and agent descriptions
- Add `evolution-audit-report.md` to the file protocol table

## No Changes To

- `plugin.json` — manifest unchanged
- `marketplace.json` — registration unchanged
- Planner/Generator/Evaluator prompts — same structure, just the inputs they read change
- `agents/product-experience-reviewer/AGENT.md` — untouched

## File Summary

| Action | File |
|--------|------|
| **NEW** | `plugins/product-loop/agents/product-evolution-reviewer/AGENT.md` |
| **EDIT** | `plugins/product-loop/skills/product-loop/SKILL.md` |
| **EDIT** | `plugins/product-loop/commands/help.md` |
