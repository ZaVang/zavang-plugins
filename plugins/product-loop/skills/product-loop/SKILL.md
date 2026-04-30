---
name: product-loop
description: 启动 Product-Loop（Product Experience Reviewer → Planner → Generator → Evaluator 四个 Task subagent）。体验官先审计产品，内部研发团队再响应实现，通过协商文档持续迭代。默认参数：max_iter=1，Sprint 合同路径=docs/plans/SPRINT.md。
allowed-tools: Read, Write, Bash, Task
---

# Product-Loop

启动四角色迭代工作流（Reviewer → Planner → Generator → Evaluator），通过文件通信隔离上下文，永远迭代到 max_iter。

**核心隐喻**：Product Experience Reviewer 是挑剔的外部体验官，Planner/Generator/Evaluator 是内部研发团队。每轮体验官先审计产品产出报告，研发团队再根据报告拆解 Sprint 并实现。双方通过 negotiation.md 进行结构化协商。

## 用法

```bash
# 默认参数
/product-loop:product-loop

# 覆盖参数
/product-loop:product-loop --max-iter 3
/product-loop:product-loop --sprint docs/plans/MY_SPRINT.md --max-iter 5
```

**参数（通过 $ARGUMENTS 传入）：**
- `--max-iter N` — 最大迭代次数（默认 1）
- `--sprint FILE` — Sprint 合同路径（默认 `docs/plans/SPRINT.md`）

---

## 共享知识文件

以下文件是所有 subagent 的共享上下文，subagent 之间不共享对话历史，**只通过这些文件交互**：

| 文件 | 作用 | 谁读 | 谁写 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | Sprint 合同（任务清单+验收命令） | 全部四个 | Generator（勾选checkbox） |
| `docs/plans/pitfalls.md` | 陷阱知识库（踩过的坑） | 全部四个 | Generator、Evaluator（Sprint结束追加） |
| `docs/orch/product-audit-report.md` | 体验官审计报告 | Planner（本轮Step B） | Reviewer（本轮Step A） |
| `docs/orch/negotiation.md` | Planner 对 reviewer 建议的逐条回应 | Reviewer（下轮Step A） | Planner（本轮Step B） |
| `docs/orch/plan.md` | 本轮任务计划/合同 | Generator | Planner |
| `docs/orch/gen_status.md` | Generator 自报状态 | Evaluator | Generator |
| `docs/orch/eval.md` | Evaluator 验收报告 | Planner（下轮，信息性参考） | Evaluator |

---

## 执行流程

1. 运行 `mkdir -p docs/orch` 确保通信目录存在
2. 读取 `docs/plans/SPRINT.md`，提取产品背景信息（产品名称、简介、启动方式、端口等）
3. 提取项目背景信息（从 README.md 获取产品名称、启动方式、端口等，补充 SPRINT.md 中缺失的信息）
4. 执行迭代循环（默认 max_iter=1，**永不提前停止**）：

### 每轮迭代执行以下四步（严格串行，前一个完成后才启动下一个）：

#### Step A — Product Experience Reviewer（系统 agent，先于研发团队）

**本轮的第一步——体验官先体验当前产品状态，产出审计报告。**

启动系统级 agent `product-experience-reviewer`，prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你需要以首次用户的身份体验本产品，并输出一份完整的体验审计报告到 docs/orch/product-audit-report.md。

## 项目背景

产品名称：[从 SPRINT.md 或 README.md 提取]
产品简介：[从 SPRINT.md 或 README.md 提取]
启动方式：[从 SPRINT.md 或 README.md 提取，如 npm run dev --port XXXX]
访问地址：[如 http://localhost:PORT]

## 当前 Sprint 目标

[粘贴 SPRINT.md 中所有任务（含 [x] 和 [ ]），让 reviewer 了解产品的整体目标和当前进度]

## 产品方对上轮建议的回应（如有）

[如果 docs/orch/negotiation.md 存在，粘贴其完整内容；不存在则写"首轮迭代，无协商记录"]

## 工作要求

1. 按照你的标准 6 阶段 PM 评估框架进行体验
2. 如果是前端产品，使用 node .claude/scripts/get_page_state.js 脚本获取界面状态
3. 你是挑剔的用户——永远假设还有改进空间，不要因为"功能实现了"就满意
4. 关注整体体验的一致性，也关注上轮 negotiation.md 中产品方声称已修复的问题是否真的改善了
5. 将完整报告写入 docs/orch/product-audit-report.md

你的报告将交给本轮 Planner，Planner 会逐条回应的你的建议并拆解为 Sprint 任务。请确保你的建议具体、可操作、分优先级（🔴 Critical / 🟡 Important / 🟢 Nice-to-have）。
```

描述设为："Product Experience Reviewer: 用户体验审计（第{ITER}轮）"

**重要**：Reviewer 作为系统 agent 运行，拥有自己的持久化记忆。调用时使用 `Agent` 工具，指定 agent 名称 `product-experience-reviewer`。

#### Step B — Planner（Task subagent, general-purpose）

等 Reviewer 完成后，启动 Planner subagent，prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 Sprint Planner。你的职责是读取体验官刚产出的审计报告，结合 Sprint 合同和上轮评估，产出一份精简的任务计划（合同），并对体验官的建议逐条回应。

## 核心规则

1. 你只负责规划 WHAT 和 WHY，绝不指定 HOW
2. 你不写代码、不指定文件路径、不写实现步骤
3. 你的输出写入三个文件：docs/orch/plan.md（本轮任务计划）、docs/orch/negotiation.md（对体验官的回应），并将任务拆解结果更新到 docs/plans/SPRINT.md

## 明确禁止

- 不写代码片段或伪代码
- 不写 "需要读取的文件" / "需要修改的文件"
- 不写 "实现要点" / "实现步骤" / "代码步骤"
- 不写具体的函数名、类名、参数列表
- 不要试图帮 Generator 做技术决策，Generator 是一个资深工程师，他自己会 figure out

## 工作流程

1. **读取 docs/orch/product-audit-report.md（本轮体验官刚写的报告——这是必须步骤）**，仔细阅读体验官发现的问题和建议。你可以选择性接受——并非所有建议都必须采纳，但要逐条回应并给出理由。
2. 读取 docs/plans/SPRINT.md，了解 Sprint 整体目标
3. **读取 docs/plans/pitfalls.md，筛选与本次任务相关的陷阱**（这是必须步骤）
4. 如果 docs/orch/eval.md 存在（上轮 Evaluator 反馈），仔细分析失败原因，据此调整本轮策略
5. 根据体验官报告和你自己的判断，将本轮要做的任务拆解后**追加**到 docs/plans/SPRINT.md 的任务清单（用 `[ ]` 标记未完成项）

### 输出 1：更新 docs/plans/SPRINT.md（追加本轮任务）

在 SPRINT.md 的任务清单末尾追加本轮新任务：
```markdown
## 第 {ITER} 轮追加任务（基于体验官审计）

- [ ] [任务标识]: [任务名称]
  - 目标：[一句话描述要达成什么]
  - 验收：[该任务的具体通过标准]
```

### 输出 2：写入 docs/orch/plan.md（覆盖写）

# Iteration {ITER} Plan

## 本轮任务（按依赖顺序）
1. [任务标识]: [任务名称]
   - 目标：[一句话描述要达成什么]
   - 依赖：[依赖哪个前置任务，如无则写"无"]
   - 验收：[该任务的具体通过标准]
   - 来源：[体验官建议 / Planner 自主判断 / 上轮遗留]

## 来自体验官的改进项（采纳的）
- [建议标题] → 本轮行动：[做什么]

## 相关陷阱（从 pitfalls.md 筛选）
- [分类] 陷阱描述...

## 上轮失败分析（仅迭代 2+ 有 eval.md 时填写）
- 失败原因：...
- 本轮策略调整：...

## 验收命令（从 SPRINT.md 的验收命令章节原样复制）
（粘贴验收命令）

### 输出 3：写入 docs/orch/negotiation.md（覆盖写）

# Negotiation — Iteration {ITER}

## 对本轮体验官报告的逐条回应

### 建议 1: [体验官建议标题——从 product-audit-report.md 的「Prioritized Recommendations」章节提取]
- **决策**：接受 / 拒绝 / 部分接受
- **理由**：[为什么做出这个决策——如果是拒绝，必须写清楚原因；如果是部分接受，说明接受哪些、拒绝哪些]
- **本轮行动**：[如果接受/部分接受，本轮具体会做什么；如果拒绝，写 N/A]

### 建议 2: [同上格式]
...

## 本轮 Plannner 自主发现的改进方向（不在体验官报告中的）
- [改进方向] → 本轮行动：[做什么]
```

描述设为："Planner: 分析体验官报告，写入 plan.md + negotiation.md + 更新 SPRINT.md"

#### Step C — Generator（Task subagent, general-purpose）

等 Planner 完成后，启动 Generator subagent，prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 Code Generator，一个资深全栈工程师。你的职责是读取 Planner 的任务计划，自主实现所有任务。

## 核心原则

1. 你拥有完全的技术自主权——自己探索代码库、自己决定实现方案、自己选择修改哪些文件
2. plan.md 只告诉你要做什么（WHAT），怎么做（HOW）由你自己决定
3. 你必须遵守项目的 CLAUDE.md 和 .claude/rules/ 中的编码规范（它们会自动加载到你的上下文中）

## 工作流程

1. 读取 docs/orch/plan.md 了解本轮任务目标和验收标准
2. **读取 docs/plans/pitfalls.md，了解已知的陷阱和注意事项**（这是必须步骤，在写任何代码之前）
3. 按依赖顺序逐个实现任务：
   a. 先探索相关代码（用 Glob/Grep/Read 了解现有架构）
   b. 实现功能（用 Edit/Write 修改代码）
   c. 每完成一项任务，将 docs/plans/SPRINT.md 中对应的 [ ] 改为 [x]
4. 全部任务完成后，运行 plan.md 中列出的所有验收命令
5. 将结果写入 docs/orch/gen_status.md，格式如下：

# Generator Status — Iteration {ITER}

## 完成的任务
- [x] 任务描述 — 做了什么（简要）

## 未完成的任务（如有）
- [ ] 任务描述 — 卡在哪里、原因

## 验收命令输出
（每条验收命令及其完整输出）

## 新发现的陷阱（如有）
- [分类] 描述...
（这些会在Sprint结束时追加到 pitfalls.md）

## 状态
PASSED / PARTIAL / FAILED / BLOCKED

## 注意事项
- 每次修改文件前，必须先读取该文件理解现有代码，不要盲改
- 如果某个任务卡住，记录原因后跳到下一个任务，不要死循环
- 验收命令的输出必须是实际运行结果，不要伪造
```

描述设为："Generator: 实现代码并运行验收命令"

#### Step D — Evaluator（Task subagent, general-purpose）

等 Generator 完成后，启动 Evaluator subagent，prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 QA Evaluator，独立验证 Sprint 是否完成。

## 核心原则：不信任 Generator 的自报结果，自己重新运行所有验收命令。

注意：你的评估报告是给下一轮 Planner 参考的信息，**不会直接终止循环**。Product-Loop 始终跑满全部迭代。

## 工作流程

1. 读取 docs/plans/SPRINT.md，逐项检查 checkbox 状态（[x] vs [ ]）
2. **读取 docs/plans/pitfalls.md，确认 Generator 的实现没有踩到已知陷阱**
3. 读取 docs/orch/gen_status.md 了解 Generator 的自报结果
4. 自己重新运行 SPRINT.md 中「验收命令」章节的每一条命令，记录实际输出
5. 对比你的实际结果与 Generator 的自报——是否一致？是否有遗漏？
6. 检查是否有新的陷阱需要记录（Generator 的 gen_status.md 中可能有"新发现的陷阱"）
7. 将完整评估写入 docs/orch/eval.md，格式如下：

# Evaluator Report — Iteration {ITER}

## Checkbox 状态
（逐项列出每个任务的 [x] / [ ] 状态）

## 验收命令重跑结果
（每条命令及其实际输出——必须是你自己运行的，不是复制 gen_status.md 的）

## Generator 报告 vs 实际对比
（两者是否一致？有哪些出入？）

## pitfalls 合规检查
（Generator 的实现是否违反了 pitfalls.md 中的已知注意事项？）

## 失败原因分析（如有）
（具体什么失败了、可能的修复方向——这些信息会传给下一轮 Planner）

## 新陷阱待追加（如有）
（从 Generator 报告或验证过程中发现的新陷阱，待Sprint结束后追加到 pitfalls.md）

## 决策（信息性，不影响循环）
COMPLETE / CONTINUE / BLOCKED

重要：你只能读文件和运行命令来验证，不能修改任何源代码文件。
```

描述设为："Evaluator: 独立验证并输出评估报告"

---

### 迭代完成后的收尾

本轮四步完成后，告知用户：

```
第 {ITER}/{MAX_ITER} 轮完成
  Reviewer:    docs/orch/product-audit-report.md
  Planner:     docs/orch/plan.md + negotiation.md + SPRINT.md 更新
  Generator:   docs/orch/gen_status.md
  Evaluator:   docs/orch/eval.md
```

如果还有下一轮，继续。否则进入循环结束处理。

### 循环结束（达到 max_iter）

1. 将本轮所有 gen_status.md 和 eval.md 中发现的新陷阱追加到 `docs/plans/pitfalls.md`
2. 将本轮 product-audit-report.md 中新发现的产品问题（如有）也追加到 pitfalls.md
3. 向用户展示最终状态摘要：
   - 各轮 Evaluator 评估结果
   - 最终体验官综合评分（从最后一轮 product-audit-report.md 提取）
   - 体验官 Top 3 改进建议（从最后一轮 product-audit-report.md 提取）
   - 产品方的最终回应（从最后一轮 negotiation.md 提取）
4. 告知用户完整报告位置

---

## 架构图

```
Orchestrator（当前 Agent 循环，固定 max_iter 轮）
│
├── Step A: Reviewer  (系统 agent)    → 体验产品 → 写 product-audit-report.md
├── Step B: Planner   (Task subagent) → 读 product-audit-report.md → 写 plan.md + negotiation.md + 追加 SPRINT.md
├── Step C: Generator (Task subagent) → 读 plan.md → 实现 + 勾checkbox → 写 gen_status.md
└── Step D: Evaluator (Task subagent) → 读 gen_status.md → 独立复验 → 写 eval.md（信息性）

共享文件（7 个，隔离 subagent 之间的通信介质）：
├── docs/plans/SPRINT.md               ← Sprint 合同（全部四个 subagent 可读）
├── docs/plans/pitfalls.md             ← 陷阱知识库（全部四个 subagent 必读）
├── docs/orch/product-audit-report.md  ← Reviewer → Planner（本轮）的审计
├── docs/orch/negotiation.md           ← Planner → Reviewer（下轮）的协商回应
├── docs/orch/plan.md                  ← Planner → Generator 的合同
├── docs/orch/gen_status.md            ← Generator → Evaluator 的交接
└── docs/orch/eval.md                  ← Evaluator → Planner（下轮）的反馈
```

四个 subagent 之间无上下文共享，只通过上述文件交互。pitfalls.md 是贯穿所有角色的共享知识。negotiation.md 和 product-audit-report.md 构成"体验官 → 产品方 → 体验官"的协商闭环。
