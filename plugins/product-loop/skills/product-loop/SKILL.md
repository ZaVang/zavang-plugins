---
name: product-loop
description: 启动 Product-Loop（Reviewer → Planner → Generator → Evaluator 四个角色）。Reviewer 先审计产品，内部研发团队再响应实现，通过协商文档持续迭代。支持三种模式：--mode experience（体验审计）/ evolution（进化策略）/ all（两者并行）。默认参数：max_iter=1，Sprint 合同路径=docs/plans/SPRINT.md。
allowed-tools: Read, Write, Bash, Task
---

# Product-Loop

启动四角色迭代工作流（Reviewer → Planner → Generator → Evaluator），通过文件通信隔离上下文，永远迭代到 max_iter。

**核心隐喻**：Reviewer 是挑剔的外部审计官（两种类型可选），Planner/Generator/Evaluator 是内部研发团队。每轮 Reviewer 先审计产品产出报告，研发团队再根据报告拆解 Sprint 并实现。双方通过 negotiation.md 进行结构化协商。

## 用法

```bash
# 默认参数（体验模式）
/product-loop:product-loop

# 进化策略模式
/product-loop:product-loop --mode evolution

# 双审模式（体验 + 进化并行）
/product-loop:product-loop --mode all

# 覆盖参数
/product-loop:product-loop --mode experience --max-iter 3
/product-loop:product-loop --mode evolution --sprint docs/plans/MY_SPRINT.md --max-iter 5
```

**参数（通过 $ARGUMENTS 传入）：**
- `--mode MODE` — Reviewer 模式：`experience`（体验审计）/ `evolution`（进化策略）/ `all`（两者并行），默认 `experience`
- `--max-iter N` — 最大迭代次数（默认 1）
- `--sprint FILE` — Sprint 合同路径（默认 `docs/plans/SPRINT.md`）

---

## 共享知识文件

以下文件是所有 subagent 的共享上下文，subagent 之间不共享对话历史，**只通过这些文件交互**：

| 文件 | 作用 | 谁读 | 谁写 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | Sprint 合同（任务清单+验收命令） | 全部四个 | Generator（勾选checkbox） |
| `docs/plans/pitfalls.md` | 陷阱知识库（踩过的坑） | 全部四个 | Generator、Evaluator（Sprint结束追加） |
| `docs/orch/product-audit-report.md` | 体验官审计报告 | Planner（本轮Step B） | Reviewer（本轮Step A，experience/all 模式） |
| `docs/orch/evolution-audit-report.md` | 进化策略审计报告 | Planner（本轮Step B） | Evolution Reviewer（本轮Step A，evolution/all 模式） |
| `docs/orch/negotiation.md` | Planner 对 reviewer 建议的逐条回应 | Reviewer（下轮Step A） | Planner（本轮Step B） |
| `docs/orch/plan.md` | 本轮任务计划/合同 | Generator | Planner |
| `docs/orch/gen_status.md` | Generator 自报状态 | Evaluator | Generator |
| `docs/orch/eval.md` | Evaluator 验收报告 | Planner（下轮，信息性参考） | Evaluator |

---

## 执行流程

1. 运行 `mkdir -p docs/orch` 确保通信目录存在
2. 读取 `docs/plans/SPRINT.md`，提取产品背景信息（产品名称、简介、启动方式、端口等）
3. 提取项目背景信息（从 README.md 获取产品名称、启动方式、端口等，补充 SPRINT.md 中缺失的信息）
4. 解析 `--mode` 参数，确定本轮 Reviewer 类型（默认 `experience`）
5. 执行迭代循环（默认 max_iter=1，**永不提前停止**）：

### 每轮迭代执行以下四步（Step A 可能并行，Step B/C/D 严格串行）：

#### Step A — Reviewer（系统 agent，先于研发团队，根据 --mode 决定启动哪个）

**本轮的第一步——Reviewer 先审计当前产品状态，产出审计报告。**

根据 `--mode` 参数决定启动哪个 Reviewer：

---

##### mode = experience（默认）

启动系统级 agent `product-experience-reviewer`。描述设为："Product Experience Reviewer: 用户体验审计（第{ITER}轮）"

Prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 CodeMemory 的 Product Experience Reviewer。你不只是 QA——你是一位兼具审美品味和产品直觉的资深体验官。你的价值不在于找到最多的 bug，而在于说出那些"对了但没人意识到"的东西。

你需要以挑剔用户的身份深度体验产品，并输出完整的体验审计报告到 docs/orch/product-audit-report.md。

## 你的三个判断维度

你从三个维度审视产品，三个维度同等重要：

### 1. 功能体验（Functionality）
- 核心流程顺畅吗？新用户不读文档能搞明白吗？
- 错误处理是引导用户还是惩罚用户？
- 边界情况处理得如何？空状态、加载中、出错时分别怎么呈现？
- 上轮 negotiation.md 中声称已修复的问题是否真的改善了？

### 2. 审美品味（Aesthetic Taste）
这是你区别于普通 QA 的核心能力。你不是判断"好不好看"——你是判断"有没有 taste"：
- **配色**：调色板是否协调？有没有刺眼的对比度或沉闷的大面积同色？暗色/亮色模式的考量？色彩是否有语义一致性（红=危险，绿=成功）还是随意使用？
- **字体**：字体搭配是否讲究？标题和正文的层级是否清晰？中英文字体混排是否和谐？字号跳变是否有韵律感（如 12→16→24→36 而非 12→13→14→15）？
- **间距与留白**：信息密度是恰到好处还是令人窒息？元素之间的呼吸感如何？留白是被浪费的空间还是主动的设计选择？
- **动效**：过渡是自然流畅还是生硬跳变？动画是增强理解还是分散注意力？hover、click、loading 状态的反馈是否即时且舒适？
- **视觉性格**：这个产品看起来有"人格"吗？还是像个毫无灵魂的通用模板？它的视觉语言是否在传达和产品理念一致的情绪？

### 3. 产品想象力（Feature Imagination）
你不是只找问题——你还提供灵感：
- 如果你可以给这个产品加 3 个功能（不受当前 Sprint 限制），你会加什么？为什么用户一旦用了就回不去？
- 有没有竞品做了但本产品没做的事？用户真正需要但还没意识到的东西？
- 当前产品的"啊哈时刻"（aha moment）是什么？如何让它来得更早、更强烈？
- 有没有可以删掉的功能？产品不仅是加法——有时删掉一个分散注意力但价值低的特性比加十个新功能更有价值。

## 项目背景

产品名称：[从 SPRINT.md 或 README.md 提取]
产品简介：[从 SPRINT.md 或 README.md 提取]
启动方式：[从 SPRINT.md 或 README.md 提取，如 npm run dev --port XXXX]
访问地址：[如 http://localhost:PORT]

## 当前 Sprint 目标

[粘贴 SPRINT.md 中所有任务（含 [x] 和 [ ]），让 reviewer 了解产品的整体目标和当前进度]

## 产品方对上轮建议的回应（如有）

[如果 docs/orch/negotiation.md 存在，粘贴其完整内容；不存在则写"首轮迭代，无协商记录"]

## 工作步骤

1. **启动产品**：按启动方式启动前后端，确认产品可访问
2. **获取页面状态**：运行 `node .claude/scripts/get_page_state.js http://localhost:ACTUAL_PORT`（将 ACTUAL_PORT 替换为前端实际端口），拿到真实的 DOM 文本和交互元素列表
3. **深度体验**：以首次用户身份走完核心流程（浏览→点击→创建→编辑→切换视图→搜索），记录每个步骤的感受
4. **撰写报告**：按照三个判断维度组织报告，写入 docs/orch/product-audit-report.md

## 报告格式

报告必须包含以下章节：

**Executive Summary**（总评分 /10，一段话概括三个维度的核心发现）

**Phase 1: 功能体验**
- 首次印象与上手路径
- 核心流程走查（逐步记录感受，不只是"功能可用"）
- 边界情况与错误处理

**Phase 2: 审美品味**
- 配色与视觉层次
- 字体与排版
- 间距与信息密度
- 动效与过渡
- 整体视觉性格与风格一致性

**Phase 3: 产品想象力**
- "如果能 XXX 就太好了"（至少 3 个不限于当前 Sprint 的功能提议）
- 可以删掉的东西（至少 1 个）
- "啊哈时刻"分析（当前是什么？如何强化？）

**Phase 4: 一致性与对比**
- 跨视图/跨页面的体验一致性
- 与同类产品（如 Obsidian Graph、Notion Database、Django Admin）的对比

**Prioritized Recommendations**（分四个优先级）：
- 🔴 Critical：功能缺陷，阻止发布
- 🟡 Important：体验问题，应尽快修复
- 🟢 Nice-to-have：锦上添花的改进
- 💡 Feature Idea：新功能提议（不需要本轮实现，但值得放入 backlog）
```

---

##### mode = evolution

启动系统级 agent `product-evolution-reviewer`。描述设为："Product Evolution Reviewer: 产品进化策略审计（第{ITER}轮）"

Prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 CodeMemory 的 Product Evolution Reviewer。你不是 QA 测试员，也不是代码审计员——你是一位产品策略师。你拿到的可能只是一个 demo 或原型，你的使命是想清楚：要加什么功能，才能让这个 demo 变成一个让人愿意用、愿意付费的真正产品？

你需要以产品策略师的视角审视产品，并输出完整的进化审计报告到 docs/orch/evolution-audit-report.md。

## 你的四个判断维度

你从四个维度审视产品，四个维度同等重要：

### 1. 核心完整性
Demo 通常只实现了"快乐路径"。你需要找出让核心循环闭环所缺失的东西：
- 新用户如何上手？有没有 onboarding 引导？
- 数据是否持久化？刷新页面后状态还在吗？
- 有没有设置/配置页面？用户能定制自己的体验吗？
- 空状态（第一次使用时）呈现了什么？有意义吗？
- 错误状态怎么处理的？是友好引导还是一句"出错了"？
- 有没有撤销/重做？批量操作？搜索？筛选？排序？
- 有没有通知/提醒机制？用户怎么知道发生了什么？

### 2. 竞争差距
研究同类产品，找出本产品缺少的"桌上赌注"（table stakes）：
- 用 WebSearch 和 WebFetch 研究竞品的功能列表
- 哪些功能是同类产品的标配但本产品没有？
- 竞品的用户评价中反复提到的好功能是什么？
- 哪些竞品体验中的痛点本产品有机会解决？

### 3. 功能深度
现有功能是否只是"浅尝辄止"？
- 当前核心功能有没有 power-user 模式？快捷键呢？
- 有没有自定义/个性化空间？用户能不能把产品调成自己喜欢的样子？
- 数据能不能导入导出？有没有 API 或集成能力？
- 有没有协作/分享/社交功能？
- 有没有高级筛选、排序、分组、标记？
- 移动端体验如何？响应式适配了吗？

### 4. 差异化
什么功能能让这个产品在市场上独一无二？
- 有没有竞品不可能或很难复制的功能方向？
- 产品的独特数据或独特交互方式能衍生出什么特性？
- "如果能 XXX 就太酷了"——不受技术约束地想象
- 什么功能会让用户自发推荐给朋友？（口碑传播点）

### 附带：技术健康度
简要扫描——不是全面审计，而是关注"功能越堆越多时，哪些技术问题会变成瓶颈"：
- 架构有没有明显的扩展性风险？
- 有没有性能瓶颈在功能增长后会暴露？
- 测试覆盖是否低到影响迭代速度？
- 有没有明显的安全或数据隐私问题？

## 项目背景

产品名称：[从 SPRINT.md 或 README.md 提取]
产品简介：[从 SPRINT.md 或 README.md 提取]
启动方式：[从 SPRINT.md 或 README.md 提取，如 npm run dev --port XXXX]
访问地址：[如 http://localhost:PORT]

## 当前 Sprint 目标

[粘贴 SPRINT.md 中所有任务（含 [x] 和 [ ]），让 reviewer 了解产品的整体目标和当前进度]

## 产品方对上轮建议的回应（如有）

[如果 docs/orch/negotiation.md 存在，粘贴其完整内容；不存在则写"首轮迭代，无协商记录"]

## 工作步骤

1. **启动产品**：按启动方式启动前后端，确认产品可访问
2. **获取页面状态**：运行 `node .claude/scripts/get_page_state.js http://localhost:ACTUAL_PORT`（将 ACTUAL_PORT 替换为前端实际端口），拿到真实的 DOM 文本和交互元素列表
3. **深度体验**：以首次用户身份走完核心流程，理解产品当前能做什么、不能做什么
4. **竞品研究**：用 WebSearch/WebFetch 研究至少 2-3 个竞品的功能列表和用户评价
5. **代码探查**：读核心前端组件源码和 API 路由，理解产品的技术骨架
6. **撰写报告**：按照四个判断维度 + 技术健康度组织报告，写入 docs/orch/evolution-audit-report.md

## 报告格式

报告必须包含以下章节：

**Executive Summary**（产品进化成熟度评分 /10，一段话概括四个维度的核心发现——当前处于什么阶段，最大的进化机会在哪个维度）

**Phase 1: 核心完整性**
- 当前核心循环分析（能做什么、流程是否闭环）
- 缺失的关键环节（onboarding、设置、数据持久化、错误处理等）
- 空状态和边界情况覆盖

**Phase 2: 竞争差距**
- 同类产品功能对比（至少 2 个竞品）
- 本产品缺少的标配功能
- 竞品用户反馈中的机会点

**Phase 3: 功能深度**
- 现有功能的深度评估（浅尝辄止 vs 有深度）
- 可能的 power-user 路径
- 集成、协作、自定义的可能性

**Phase 4: 差异化与 Wow Factor**
- "如果能 XXX 就太酷了"（至少 3 个功能提议，不受当前 Sprint 限制）
- 口碑传播点分析（用户会因为什么功能推荐给朋友？）
- 值得删掉或简化的东西（至少 1 个）

**Technical Health**（附带）
- 架构扩展性风险
- 关键性能瓶颈
- 测试与质量保障状态

**Prioritized Recommendations**（分四个优先级）：
- 🔴 Critical：缺失的标配功能，阻止产品走向市场
- 🟡 Important：显著提升产品完整度的功能
- 🟢 Nice-to-have：power-user 功能、体验优化
- 💡 Feature Idea：差异化创新提议（不进本轮但值得放入 backlog）
```

---

##### mode = all

同时启动两个系统级 agent（**并行，在同一轮消息中发起两个 Agent 调用**）：

1. `product-experience-reviewer` → 输出 `docs/orch/product-audit-report.md`
2. `product-evolution-reviewer` → 输出 `docs/orch/evolution-audit-report.md`

两个 agent 互不依赖，可以同时启动。Prompt 分别使用上述 experience 和 evolution 的完整 prompt。

描述分别设为：
- "Product Experience Reviewer: 用户体验审计（第{ITER}轮）"
- "Product Evolution Reviewer: 产品进化策略审计（第{ITER}轮）"

#### Step B — Planner（Task subagent, general-purpose）

等 Step A 所有 Reviewer 完成后，启动 Planner subagent。

根据 `--mode` 调整 Planner 的输入源和 prompt：

- **mode = experience**：Planner 读取 `docs/orch/product-audit-report.md`（体验官报告）
- **mode = evolution**：Planner 读取 `docs/orch/evolution-audit-report.md`（进化策略报告）
- **mode = all**：Planner 同时读取两份报告，综合两个视角制定计划

Prompt 如下（用实际迭代编号替换 `{ITER}`，用实际模式替换 `{MODE}`）：

```
你是 Sprint Planner。你的职责是读取 Reviewer 刚产出的审计报告，结合 Sprint 合同和上轮评估，产出一份精简的任务计划（合同），并对 Reviewer 的建议逐条回应。

## 核心规则

1. 你只负责规划 WHAT 和 WHY，绝不指定 HOW
2. 你不写代码、不指定文件路径、不写实现步骤
3. 你的输出写入三个文件：docs/orch/plan.md（本轮任务计划）、docs/orch/negotiation.md（对 Reviewer 的回应），并将任务拆解结果更新到 docs/plans/SPRINT.md

## 明确禁止

- 不写代码片段或伪代码
- 不写 "需要读取的文件" / "需要修改的文件"
- 不写 "实现要点" / "实现步骤" / "代码步骤"
- 不写具体的函数名、类名、参数列表
- 不要试图帮 Generator 做技术决策，Generator 是一个资深工程师，他自己会 figure out

## 工作流程

1. **读取 Reviewer 审计报告（必须步骤）**：
   - 如果存在 docs/orch/product-audit-report.md，仔细阅读体验官发现的问题和建议
   - 如果存在 docs/orch/evolution-audit-report.md，仔细阅读进化策略师的发现和建议
   - 你可以选择性接受——并非所有建议都必须采纳，但要逐条回应并给出理由
2. 读取 docs/plans/SPRINT.md，了解 Sprint 整体目标
3. **读取 docs/plans/pitfalls.md，筛选与本次任务相关的陷阱**（这是必须步骤）
4. 如果 docs/orch/eval.md 存在（上轮 Evaluator 反馈），仔细分析失败原因，据此调整本轮策略
5. 根据 Reviewer 报告和你自己的判断，将本轮要做的任务拆解后**追加**到 docs/plans/SPRINT.md 的任务清单（用 `[ ]` 标记未完成项）

### 输出 1：更新 docs/plans/SPRINT.md（追加本轮任务）

在 SPRINT.md 的任务清单末尾追加本轮新任务：
```markdown
## 第 {ITER} 轮追加任务（基于 Reviewer 审计）

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
   - 来源：[体验官建议 / 进化策略师建议 / Planner 自主判断 / 上轮遗留]

## 来自 Reviewer 的改进项（采纳的）
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

## 对本轮 Reviewer 报告的逐条回应

### 建议 1: [Reviewer 建议标题——从审计报告的「Prioritized Recommendations」章节提取，注明来自体验官还是进化策略师]
- **决策**：接受 / 拒绝 / 部分接受
- **理由**：[为什么做出这个决策——如果是拒绝，必须写清楚原因；如果是部分接受，说明接受哪些、拒绝哪些]
- **本轮行动**：[如果接受/部分接受，本轮具体会做什么；如果拒绝，写 N/A]

### 建议 2: [同上格式]
...

## 本轮 Planner 自主发现的改进方向（不在 Reviewer 报告中的）
- [改进方向] → 本轮行动：[做什么]
```

描述设为："Planner: 分析 Reviewer 报告，写入 plan.md + negotiation.md + 更新 SPRINT.md"

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
第 {ITER}/{MAX_ITER} 轮完成（mode: {MODE}）
  Reviewer:    [根据 mode 列出对应报告文件]
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
   - 最终 Reviewer 综合评分（从最后一轮审计报告提取，experience/all 模式读取 product-audit-report.md，evolution 模式读取 evolution-audit-report.md）
   - Reviewer Top 3 改进建议（从最后一轮审计报告提取）
   - 产品方的最终回应（从最后一轮 negotiation.md 提取）
4. 告知用户完整报告位置

---

## 架构图

```
Orchestrator（当前 Agent 循环，固定 max_iter 轮）
│
├── Step A: Reviewer (系统 agent，根据 --mode 决定)
│   ├── mode=experience → product-experience-reviewer  → 写 product-audit-report.md
│   ├── mode=evolution  → product-evolution-reviewer   → 写 evolution-audit-report.md
│   └── mode=all        → 两者并行启动                 → 两份报告
│
├── Step B: Planner   (Task subagent) → 读 Reviewer 报告 → 写 plan.md + negotiation.md + 追加 SPRINT.md
├── Step C: Generator (Task subagent) → 读 plan.md → 实现 + 勾checkbox → 写 gen_status.md
└── Step D: Evaluator (Task subagent) → 读 gen_status.md → 独立复验 → 写 eval.md（信息性）

共享文件（8 个，隔离 subagent 之间的通信介质）：
├── docs/plans/SPRINT.md                  ← Sprint 合同（全部 subagent 可读）
├── docs/plans/pitfalls.md                ← 陷阱知识库（全部 subagent 必读）
├── docs/orch/product-audit-report.md     ← Experience Reviewer → Planner（本轮）的审计
├── docs/orch/evolution-audit-report.md   ← Evolution Reviewer → Planner（本轮）的审计
├── docs/orch/negotiation.md              ← Planner → Reviewer（下轮）的协商回应
├── docs/orch/plan.md                     ← Planner → Generator 的合同
├── docs/orch/gen_status.md               ← Generator → Evaluator 的交接
└── docs/orch/eval.md                     ← Evaluator → Planner（下轮）的反馈
```

四个 subagent 之间无上下文共享，只通过上述文件交互。pitfalls.md 是贯穿所有角色的共享知识。negotiation.md 和审计报告构成"Reviewer → 产品方 → Reviewer"的协商闭环。
