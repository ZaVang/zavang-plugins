在运行 `/product-loop:product-loop` 之前，请先阅读此指南，确认所有前置条件就绪。

---

## 什么是 Product-Loop？

一个统一的**两层迭代引擎**，五个阶段：**Reviewer → Scout → Planner → Generator → Evaluator**。

```
Orchestrator（你当前的 Claude 会话）
 ├── Step A  Reviewer  → 审计产品 → 写审计报告      ── Tier 1（--tier1 门控）
 │   ├── experience / evolution / research / all（--mode 决定）
 │   （--tier1 off：跳过，人写的 SPRINT.md 当需求源 = multi-ralph 行为）
 ├── Step B  Scout     → 接地代码 → 写 scout.md     ── Tier 2（--scout 门控，只读）
 ├── Step C  Planner   → 读审计报告/SPRINT + scout约束段 → 拆解 Sprint（tier1 on 时逐条回应）
 ├── Step D  Generator → 读计划 + scout代码地图段 → 自主实现 → 勾 checkbox → 写状态报告
 └── Step E  Evaluator → 独立重跑验收命令 → 写评估报告（tier1 off 时 +DECISION 控停）
```

各 subagent **互不共享上下文**，只通过文件通信。两个核心旋钮：
- **`--tier1 on|off`**（默认 on）：开 = Reviewer 自动批判产品、生成需求；关 = 人写的 SPRINT 当需求源（multi-ralph 行为）。终止也跟随它：on 跑满轮次，off 目标驱动停。
- **`--scout on|off`**（默认 on）：开 = Scout 先接地代码现状，把"约束"喂 Planner、"代码地图"喂 Generator。

Tier1 on 时 Reviewer 与产品方通过 negotiation.md 结构化协商。

**三种 Reviewer**：
- **Product Experience Reviewer**（体验官）：挑剔的外部体验者，从功能体验、审美品味、产品想象力审视产品
- **Product Evolution Reviewer**（进化策略师）：产品策略师，从核心完整性、竞争差距、功能深度、差异化提出功能进化方案
- **Product Research Reviewer**（研究员）：设计研究员，从核心假设质疑、相邻领域研究、逻辑完备性、替代设计提案提出更好的设计方向

---

## ⚠️ 前置条件

### 必须存在：`docs/plans/SPRINT.md`

至少包含产品基本信息和验收命令。任务清单可以很简洁，Planner 会在首轮基于体验官报告追加任务。

```markdown
# Sprint: 产品名称

## 产品信息
- 名称：XXX
- 简介：XXX
- 启动方式：npm run dev --port 5173
- 访问地址：http://localhost:5173

## 任务清单

- [ ] 初始任务（可选，Planner 会自动追加）

## 验收命令

```bash
npm run build
npm run lint
```
```

> 以下 Reviewer agent **仅 `--tier1 on` 时需要**。若用 `--tier1 off`（multi-ralph 行为），无需任何 Reviewer，SPRINT.md 里的 `[ ]` 任务即需求源。

### 必须存在（仅 tier1 on）：系统级 agent `product-experience-reviewer`

通过 `/agents` 创建。experience / all 模式需要。

### 必须存在（evolution / all 模式）：系统级 agent `product-evolution-reviewer`

通过 `/agents` 创建。evolution / all 模式需要。

### 必须存在（research / all 模式）：系统级 agent `product-research-reviewer`

通过 `/agents` 创建。research / all 模式需要。

### 建议存在：`docs/plans/pitfalls.md`

初建可留空，后续自动追加。

### 建议存在：`docs/project_structure.md`

持久的全库文件落点地图。**有它则**：Scout 拿它当导航起点（更快更准）、Generator 改代码后同步更新它、Evaluator 守它防漂移。缺失即优雅跳过（三个角色各自照常工作，只是少了这层加速与防腐）。建议手动种一次初始版本（每个文件/目录是干嘛的），之后由 loop 维护。

---

## 快速开始

```
# 默认：tier1 on + 体验模式 + scout on + 跑满轮次
/product-loop:product-loop

# 完全体：三审并行 + 接地
/product-loop:product-loop --mode all

# 进化策略 / 设计研究
/product-loop:product-loop --mode evolution
/product-loop:product-loop --mode research

# multi-ralph 行为：关 Tier1，人写的 SPRINT 当需求源，目标驱动停
/product-loop:product-loop --tier1 off

# 关 Scout（greenfield / 小 sprint，省一个只读 agent）
/product-loop:product-loop --mode evolution --scout off

# 参数组合
/product-loop:product-loop --mode experience --max-iter 3
/product-loop:product-loop --tier1 off --sprint docs/plans/MY_SPRINT.md --max-iter 5
```

**两个 preset = 参数组合：**
- `multi-ralph 行为` = `--tier1 off [--scout on]`（人给 SPRINT，Scout→P→G→E，COMPLETE 即停）
- `product-loop 完全体` = `--tier1 on --mode all --scout on`（Reviewer→Scout→P→G→E，跑满轮次）

---

## 两个 preset 的区别（同一引擎的两个档位）

product-loop 现在是统一引擎；`--tier1` 旋钮决定它表现为哪种：

| | `--tier1 off`（multi-ralph 行为） | `--tier1 on`（默认，产品进化） |
|---|---|---|
| 需求源（What） | 人写的 SPRINT.md | Reviewer 自动批判生成 |
| 第一步 | Scout 接地 / Planner 读 Sprint | Reviewer 审计产品 |
| 停止条件 | Evaluator `DECISION: COMPLETE` 即停（目标驱动） | 跑满 max_iter（持续进化） |
| 外部视角 | 无 | Reviewer 独立审计（experience/evolution/research/all） |
| 协商机制 | 无 | negotiation.md 闭环 |
| SPRINT.md | 用户预先写好完整任务 | Planner 每轮基于 Reviewer 报告追加任务 |
| Scout | 可选（`--scout`，默认 on） | 可选（`--scout`，默认 on） |

> 注：独立的 `multi-ralph` 插件仍保留为最精简的纯执行循环；`product-loop --tier1 off` 在其基础上多了可选的 Scout 接地阶段。

---

## 文件协议

| 文件 | 谁写 | 谁读 | 作用 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | 用户 + Planner + Generator | 全部角色 | Sprint 合同 |
| `docs/plans/pitfalls.md` | G + E（追加） | 全部角色 | 陷阱知识库 |
| `docs/project_structure.md` | Generator（改代码后同步） | Scout（导航）/Generator | 持久代码地图防漂移；Evaluator 守 |
| `docs/orch/product-audit-report.md` | Experience Reviewer（Step A） | Scout/Planner | 体验官审计报告（仅 tier1 on） |
| `docs/orch/evolution-audit-report.md` | Evolution Reviewer（Step A） | Scout/Planner | 进化策略审计报告（仅 tier1 on） |
| `docs/orch/research-audit-report.md` | Research Reviewer（Step A） | Scout/Planner | 设计研究报告（仅 tier1 on） |
| `docs/orch/scout.md` | Scout（Step B） | Planner(约束段)/Generator(代码地图段) | 代码接地（仅 scout on） |
| `docs/orch/negotiation.md` | Planner（Step C） | Reviewer（下轮 Step A） | 产品方逐条回应（仅 tier1 on） |
| `docs/orch/plan.md` | Planner（Step C） | Generator（Step D） | 本轮任务计划 |
| `docs/orch/gen_status.md` | Generator（Step D） | Evaluator（Step E） | 实现结果自报 |
| `docs/orch/eval.md` | Evaluator（Step E） | Planner（下轮） | 验收报告（tier1 off 时附 DECISION） |
