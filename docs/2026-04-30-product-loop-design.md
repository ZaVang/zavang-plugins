# Product-Loop 设计文档

**日期**: 2026-04-30
**状态**: 待实现

---

## 概述

Product-Loop 是 multi-ralph 的增强版，在内部研发循环（Planner → Generator → Evaluator）外层加入独立的 Product Experience Reviewer agent，形成"产品方研发 → 体验官审计 → 产品方改进"的持续迭代闭环。

核心隐喻：**产品方是内部研发团队，体验官是挑剔的外部用户。双方通过结构化文档协商，不共享上下文。**

---

## 与 multi-ralph 的关键区别

| | multi-ralph | product-loop |
|---|---|---|
| 停止条件 | Evaluator COMPLETE 即提前结束 | 永远跑满 max_iter |
| 默认 max_iter | 2 | 1 |
| 角色数 | 3（P/G/E） | 4（P/G/E/R） |
| 外部视角 | 无 | 体验官独立审计 |
| Evaluator 决策 | 决定流程走向 | 仅信息性，供 Planner 参考 |
| 协商机制 | 无 | negotiation.md 闭环 |

---

## 架构

```
Orchestrator（当前 session 循环，固定 max_iter 轮）
│
├── Step A: Planner
│   读 → SPRINT.md + pitfalls.md + product-audit-report.md(上轮)
│   写 → plan.md + negotiation.md
│
├── Step B: Generator
│   读 → plan.md + pitfalls.md
│   写 → gen_status.md + 勾 SPRINT.md checkbox
│
├── Step C: Evaluator（信息性）
│   读 → gen_status.md + pitfalls.md
│   写 → eval.md（含 DECISION，不影响流程）
│
└── Step D: Product Experience Reviewer
    读 → product-audit-report.md(上轮) + negotiation.md(上轮)
    体验产品 → 运行脚本、操作界面
    写 → product-audit-report.md
```

---

## 文件协议（7 个共享文件）

| 文件 | 谁写 | 谁读 | 作用 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | 用户（手动）+ Generator（勾checkbox） | P, G, E | Sprint 合同 |
| `docs/plans/pitfalls.md` | Generator + Evaluator（追加） | P, G, E, R | 陷阱知识库 |
| `docs/orch/plan.md` | Planner | Generator | 本轮任务计划 |
| `docs/orch/gen_status.md` | Generator | Evaluator | 实现结果自报 |
| `docs/orch/eval.md` | Evaluator | Planner（下轮） | 验收报告（信息性） |
| `docs/orch/product-audit-report.md` | Reviewer | Planner（下轮）+ Reviewer（下轮） | 体验官审计报告 |
| `docs/orch/negotiation.md` | Planner | Reviewer（下轮） | 产品方对建议的逐条回应 |

---

## 四角色 Prompt 设计

### Step A — Planner

在 multi-ralph Planner 基础上增加：
- 读取上轮 `product-audit-report.md`（如有）
- 读取上轮 `negotiation.md`（如有）
- 对 reviewer 建议逐条回应，写入 `negotiation.md`

negotiation.md 格式：
```markdown
# Negotiation — Iteration {ITER}

## 对上轮体验官报告的回应

### 建议 1: [建议标题]
- 决策：接受 / 拒绝 / 部分接受
- 理由：[为什么]
- 本轮行动：[如果接受/部分接受，本轮会做什么]

### 建议 2: ...
```

### Step B — Generator

与 multi-ralph Generator 相同。无需感知 reviewer 的存在。

### Step C — Evaluator

与 multi-ralph Evaluator 相同，仍输出 DECISION，但 orchestrator 不据此停止循环——仅作为 Planner 下轮参考信息。

### Step D — Product Experience Reviewer

通过 `Agent` 工具调用系统级 agent `product-experience-reviewer`。额外注入：
- 当前迭代编号
- 上轮 `product-audit-report.md`（如有，了解自己之前的反馈）
- 上轮 `negotiation.md`（如有，了解产品方回应）
- 项目背景：产品名称、启动方式、端口
- 写入路径：`docs/orch/product-audit-report.md`

Reviewer 使用内置的 6 阶段 PM 评估框架独立工作。前端产品可使用 `get_page_state.js` 脚本获取界面状态。

---

## 停止条件

**唯一停止条件：达到 max_iter。** 不存在 COMPLETE 提前停止。

达到 max_iter 后：
- 汇报最终状态（eval.md + product-audit-report.md 摘要）
- 将新陷阱追加到 pitfalls.md

---

## 参数

- `--max-iter N` — 最大迭代次数（默认 1）
- `--sprint FILE` — Sprint 合同路径（默认 `docs/plans/SPRINT.md`）

---

## 实现

新建 Claude Code plugin `product-loop`，与 multi-ralph 同仓库（zavang-plugins），结构：

```
product-loop/
├── .claude-plugin/plugin.json
├── commands/help.md
└── skills/product-loop/SKILL.md
```

SKILL.md 包含完整编排逻辑，通过 `Agent` 工具串行调用四个 subagent。
