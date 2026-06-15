---
name: product-loop
description: 启动统一的两层迭代引擎（Reviewer → Scout → Planner → Generator → Evaluator 五个阶段）。Tier1（Reviewer）通过 --tier1 开关门控：开 = 产品侧自动批判生成需求；关 = 人写的 SPRINT 当需求源（即 multi-ralph 行为）。Tier2（Scout→P→G→E）始终执行，Scout 负责代码接地。Reviewer 支持四种模式：--mode experience（体验审计）/ evolution（进化策略，兼竞品研究）/ research（设计研究）/ all（三者并行）。终止跟随 Tier1 自动切换：tier1 off=目标驱动停，tier1 on=跑满轮次。默认参数：tier1=on，scout=on，max_iter=1，Sprint 合同路径=docs/plans/SPRINT.md。
allowed-tools: Read, Write, Bash, Task
---

# Product-Loop — 统一两层迭代引擎

启动一个两层迭代工作流，通过文件通信隔离上下文：

- **Tier 1 — 产品方向（Reviewer，`--tier1` 门控）**：自动批判产品、生成"要做什么"。关掉它，则由人写的 SPRINT 当需求源。
- **Tier 2 — 研发执行（Scout → Planner → Generator → Evaluator，始终执行）**：Scout 接地代码现状，P/G/E 规划→实现→独立验收。

完整阶段顺序：**Reviewer → Scout → Planner → Generator → Evaluator**。

**核心隐喻**：Reviewer 是挑剔的外部审计官 / 自动 PM（三种镜头可选），Scout 是连接需求与代码的侦察兵（接地"怎么实现/能不能做"），Planner/Generator/Evaluator 是内部研发团队。Tier1 开时双方通过 negotiation.md 结构化协商；Tier1 关时引擎退化为 multi-ralph 式的纯执行循环。

## 用法

```bash
# 默认（Tier1 on，体验模式，Scout on，跑满轮次）
/product-loop:product-loop

# 完全体：三审并行 + 接地
/product-loop:product-loop --mode all

# 进化策略 / 设计研究
/product-loop:product-loop --mode evolution
/product-loop:product-loop --mode research

# multi-ralph 行为：关掉 Tier1，人写的 SPRINT 当需求源，目标驱动停
/product-loop:product-loop --tier1 off

# 关掉 Scout（greenfield / 小 sprint，省一个只读 agent）
/product-loop:product-loop --mode evolution --scout off

# 覆盖参数
/product-loop:product-loop --mode experience --max-iter 3
/product-loop:product-loop --tier1 off --sprint docs/plans/MY_SPRINT.md --max-iter 5
```

**参数（通过 $ARGUMENTS 传入）：**
- `--tier1 on|off` — Reviewer 段（Tier 1）开关。默认 `on`。`off` = 跳过 Reviewer，人写的 SPRINT.md 当需求源（multi-ralph 行为）
- `--mode MODE` — Reviewer 镜头：`experience`（体验审计）/ `evolution`（进化策略，兼竞品研究）/ `research`（设计研究）/ `all`（三者并行），默认 `experience`。**仅 `--tier1 on` 时生效**
- `--scout on|off` — Scout 代码接地段开关。默认 `on`
- `--max-iter N` — 最大迭代次数（默认 1）
- `--sprint FILE` — Sprint 合同路径（默认 `docs/plans/SPRINT.md`）

**终止语义（自动跟随 `--tier1`）：**
- `--tier1 off`：**目标驱动**——Evaluator 输出 `DECISION: COMPLETE` 即停（同 multi-ralph）
- `--tier1 on`：**跑满 max_iter**——Evaluator 决策仅信息性，产品持续进化

**两个 preset = 参数组合：**
```
multi-ralph 行为   = --tier1 off [--scout on]
product-loop 完全体 = --tier1 on --mode all --scout on
```

---

## 共享知识文件

以下文件是所有 subagent 的共享上下文，subagent 之间不共享对话历史，**只通过这些文件交互**：

| 文件 | 作用 | 谁读 | 谁写 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | Sprint 合同（任务清单+验收命令） | 全部角色 | Generator（勾选checkbox）；Tier1 关时由 Planner 直接读 |
| `docs/plans/pitfalls.md` | 陷阱知识库（踩过的坑） | 全部角色 | Generator、Evaluator（Sprint结束追加） |
| `docs/project_structure.md` | **持久全库文件落点地图（防漂移）** | Scout（导航起点）、Generator | **Generator（改代码后同步更新）**；Evaluator 守它防漂移 |
| `docs/orch/product-audit-report.md` | 体验官审计报告 | Scout + Planner（本轮） | Reviewer（experience/all 模式，**仅 Tier1 on**） |
| `docs/orch/evolution-audit-report.md` | 进化策略审计报告 | Scout + Planner（本轮） | Evolution Reviewer（evolution/all 模式，**仅 Tier1 on**） |
| `docs/orch/research-audit-report.md` | 设计研究报告 | Scout + Planner（本轮） | Research Reviewer（research/all 模式，**仅 Tier1 on**） |
| `docs/orch/scout.md` | **代码接地：约束/可行性段（→Planner）+ 代码地图/坑段（→Generator）** | Planner、Generator（本轮） | **Scout（本轮Step B，仅 --scout on）** |
| `docs/orch/negotiation.md` | Planner 对 reviewer 建议的逐条回应 | Reviewer（下轮Step A） | Planner（本轮Step C，**仅 Tier1 on**） |
| `docs/orch/plan.md` | 本轮任务计划/合同 | Generator | Planner |
| `docs/orch/gen_status.md` | Generator 自报状态 | Evaluator | Generator |
| `docs/orch/eval.md` | Evaluator 验收报告 | Planner（下轮，信息性参考） | Evaluator |

---

## 执行流程

1. 运行 `mkdir -p docs/orch` 确保通信目录存在
2. 读取 `docs/plans/SPRINT.md`，提取产品背景信息（产品名称、简介、启动方式、端口等）
3. 提取项目背景信息（从 README.md 获取产品名称、启动方式、端口等，补充 SPRINT.md 中缺失的信息）
4. 解析参数：`--tier1`（默认 `on`）、`--mode`（默认 `experience`，仅 tier1 on 生效）、`--scout`（默认 `on`）、`--max-iter`、`--sprint`
5. 执行迭代循环（默认 max_iter=1）。**终止跟随 `--tier1`**：
   - `--tier1 on`：跑满 max_iter（产品持续进化，Evaluator 决策仅信息性）
   - `--tier1 off`：目标驱动——Evaluator 输出 `DECISION: COMPLETE` 即停（同 multi-ralph）

### 每轮迭代执行以下阶段（Step A 内 mode=all 时三 reviewer 并行；Step A 受 --tier1、Step B 受 --scout 各自可跳过；Step C/D/E 严格串行）：

**阶段总览**：Step A Reviewer（Tier1 门控）→ Step B Scout（可选接地）→ Step C Planner → Step D Generator → Step E Evaluator。

#### Step A — Reviewer（系统 agent，**仅 `--tier1 on` 时执行**，根据 --mode 决定启动哪个）

**若 `--tier1 off`：跳过整个 Step A**，不启动任何 Reviewer，也不产出审计报告。本轮"要做什么"直接来自人写的 `docs/plans/SPRINT.md` 里的 `[ ]` 未完成任务（multi-ralph 行为）。直接进入 Step B。

**若 `--tier1 on`（默认）——本轮第一步：Reviewer 先审计当前产品状态，产出审计报告。**

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

##### mode = research

启动系统级 agent `product-research-reviewer`。描述设为："Product Research Reviewer: 设计研究审计（第{ITER}轮）"

Prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 CodeMemory 的 Product Research Reviewer。你不是产品经理，也不是代码审计员——你是一位设计研究员。你关注的是产品的核心设计逻辑是否足够好、是否有被忽略的替代方案、相邻领域是否有更优雅的解法可以借鉴。你的使命不是找问题，而是回答一个问题："如果重新思考这件事，有没有更聪明的方式？"

你需要以设计研究员的视角审视产品，并输出完整的设计研究报告到 docs/orch/research-audit-report.md。

## 你的四个判断维度

你从四个维度审视产品，四个维度同等重要：

### 1. 核心假设质疑（Assumption Questioning）
挖出产品的核心设计假设，逐一审视：
- **隐喻假设**：当前产品用什么隐喻理解问题？（如"记忆 = 文件"、"记忆加载 = 依赖解析"、"遗忘 = 不可达"）这个隐喻是否足够好？有没有被这个隐喻限制住了思维？
- **边界假设**：哪些场景被认为是"不需要处理的"？这些假设还成立吗？如果边界扩大一倍，设计还站得住吗？
- **用户假设**：假设用户是怎样的？（技术人员？AI agent？两者？）如果用户画像变化，设计的哪些部分会最先崩塌？
- **技术假设**：当前方案依赖什么技术前提？（文件系统、YAML、DAG）有没有更合适的技术底座？

### 2. 相邻领域研究（Adjacent Domain Research）
你不是在真空中思考——大量好方案已经被其他领域验证过：
- **知识图谱**：三元组存储、本体建模、推理引擎——哪些思路可以迁移到记忆管理？
- **认知架构**（ACT-R、SOAR、Global Workspace Theory）：人类记忆如何工作？间隔重复、遗忘曲线、工作记忆 vs 长期记忆——能否映射到 AI 记忆系统？
- **笔记工具**（Obsidian、Notion、Roam Research、Logseq）：双向链接、块引用、图谱视图——它们的交互范式能否迁移？
- **版本控制系统**（Git）：commit、branch、merge、rebase、tag——记忆版本管理是否可以借鉴这些概念？
- **编程语言设计**：类型系统、trait/interface、继承 vs 组合——对记忆 schema 设计有何启发？

### 3. 逻辑完备性（Logical Completeness）
审视当前设计的内在一致性：
- **概念闭环**：核心概念（atom、schema、imports、stale、maturity、intensity）之间是否形成了自洽的逻辑体系？有没有概念之间存在模糊地带？
- **边界完备**：极端场景下设计是否还能成立？0 个记忆、10000 个记忆、全是循环依赖、全是孤立节点、全标记为 stale——这些情况分别成立吗？
- **操作完备**：CRUD + 检索 + 分析 这组操作是否覆盖全部需求？有什么操作是用户（Agent）想做但做不了的？
- **演化路径**：当前设计假设的演化方向是什么？如果加入新概念（如"关联强度"、"时效性衰减"），现有模型能否承载？

### 4. 替代设计提案（Alternative Design Proposals）
不是给修补建议，而是提出"如果换一种思路会怎样"：
- **核心机制替代**：对于每个核心机制（记忆加载、遗忘处理、依赖解析、token 裁剪），提出 1-2 种本质上不同的替代方案
- **概念重组**：有没有可能合并或拆分某些概念？（如 atom + schema 是否有更好的统一方式？）
- **tradeoff 分析**：每种替代方案在简洁性、可理解性、可扩展性、性能之间做了什么 tradeoff？什么时候值得换？
- **灵感提案**：不受当前架构约束，"如果能做一件事让整个系统看起来完全不同，那会是什么？"

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

1. **深度理解现有设计**：读 CLAUDE.md、核心模块源码（resolve.py、validate.py、models.py、index.py 等）、README.md，确保完全理解设计意图和核心机制
2. **挖出隐含假设**：列出系统中所有"被视为理所当然"的设计决策，写成假设清单
3. **启动产品体验**：跑起来，用 CLI 命令（resolve、focus、wander、overview 等）实际体验，理解用户视角
4. **相邻领域研究**：用 WebSearch/WebFetch 研究至少 3 个相邻领域的设计模式——不要只搜"memory management for AI"，要搜"knowledge graph storage patterns"、"human memory models in cognitive architecture"、"note-taking app linking mechanisms"等
5. **脑力激荡**：基于研究和假设质疑，列出所有"也许可以试试"的方向，不做筛选
6. **收敛与提案**：从脑力激荡中选出最有价值的 3-5 个方向，做 tradeoff 分析
7. **撰写报告**：按照四个判断维度组织报告，写入 docs/orch/research-audit-report.md

## 报告格式

报告必须包含以下章节：

**Executive Summary**
- 核心假设概述（当前设计基于哪 3-5 个关键假设）
- 最大的研究发现在哪个维度
- 一句话：如果重新思考这件事，最有价值的突破方向是什么

**Phase 1: 核心假设质疑**
- **假设清单**：列出所有识别到的隐含假设（至少 5 个），标注哪些是可质疑的
- **关键假设深挖**：选 2-3 个最重要的假设，逐一分析——如果假设不成立，设计会怎么变？
- **被隐喻限制的地方**：当前隐喻（如"记忆 = 文件"）在哪些场景下变得牵强？

**Phase 2: 相邻领域研究**
- **领域扫描**：列出研究过的领域，每个领域简述核心思想和可迁移的点
- **可迁移模式**：选出 3-5 个最值得借鉴的设计模式，说明为什么适合本产品
- **竞品设计分析**：如果有同类产品，分析它们的设计取舍（不是功能对比，是设计哲学对比）

**Phase 3: 逻辑完备性**
- **概念体系评估**：当前核心概念之间的关系图（文字描述），标注模糊地带
- **极端场景检验**：选 3 个极端场景，验证设计是否成立
- **操作缺口**：用户（Agent）想做但做不到的事情
- **演化瓶颈**：如果要加新能力，当前模型哪里会卡住？

**Phase 4: 替代设计提案**
- **核心机制替代方案**：对 2-3 个核心机制提出本质上不同的设计
- **概念重组方案**：有没有更好的概念组织方式？
- **Tradeoff 矩阵**：每个替代方案 vs 现状的 tradeoff 分析
- **灵感炸弹**："如果能做一件事让整个系统看起来完全不同"（至少 2 个）

**Prioritized Research Directions**（分四个优先级）：
- 🔴 High-impact, Low-effort：小改动大收益的设计调整（本轮可以考虑）
- 🟡 High-impact, High-effort：值得深入但需要较大重构（放入 backlog）
- 🟢 Thought-provoking：有趣但不紧急的方向（长期研究）
- 💡 Wild idea：脑洞大开，可能不靠谱但也可能改变游戏规则
```

---

##### mode = all

同时启动三个系统级 agent（**并行，在同一轮消息中发起三个 Agent 调用**）：

1. `product-experience-reviewer` → 输出 `docs/orch/product-audit-report.md`
2. `product-evolution-reviewer` → 输出 `docs/orch/evolution-audit-report.md`
3. `product-research-reviewer` → 输出 `docs/orch/research-audit-report.md`

三个 agent 互不依赖，可以同时启动。Prompt 分别使用上述 experience、evolution 和 research 的完整 prompt。

描述分别设为：
- "Product Experience Reviewer: 用户体验审计（第{ITER}轮）"
- "Product Evolution Reviewer: 产品进化策略审计（第{ITER}轮）"
- "Product Research Reviewer: 设计研究审计（第{ITER}轮）"

#### Step B — Scout（Task subagent, general-purpose，**仅 `--scout on` 时执行**）

**若 `--scout off`：跳过整个 Step B**，不产出 scout.md，直接进入 Step C（Planner）。Planner / Generator 各自像以前一样自行探索代码。

**若 `--scout on`（默认）——代码接地侦察。** Scout 是连接"要做什么"和"代码现状"的桥梁：它**只读不改**，先摸清当前代码实际怎么跑、本轮方向触及哪些地方、有哪些约束和坑，产出 `docs/orch/scout.md`，分两段分别喂给 Planner（约束）和 Generator（代码地图）。

等 Step A 完成后（或 tier1 off 时直接），启动 Scout subagent，prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 Code Scout，研发团队的侦察兵。你不规划、不实现、不评判——你只回答一个问题：
"针对本轮要做的方向，当前代码库的现实是什么？哪些能做、哪些有约束、要动哪里、有什么坑？"

## 核心原则

1. 你只读不写源码——工具仅限 Read / Glob / Grep。你唯一的产出是写入 docs/orch/scout.md。
2. 你是 targeted 侦察，不是全库普查：只接地"本轮方向"涉及的部分，别试图理解整个代码库。
3. 你的产出分两个受众：约束段给 Planner（影响 WHAT），代码地图段给 Generator（HOW 接地）。
4. **先看地图再探路**：若存在 docs/project_structure.md（持久的全库文件落点地图），先读它定位本轮方向涉及哪些目录/文件，再针对性深入——不要从零摸整个库。它是你的导航起点，scout.md 的「相关文件」应与它一致（若你发现它与实际代码不符，在对应方向的 B 段「注意的坑」里记一条漂移，供 Generator 修正 project_structure.md）。

## 本轮方向来源

- 如果存在审计报告（docs/orch/product-audit-report.md / evolution-audit-report.md /
  research-audit-report.md 中任意一个或多个，Tier1 on 时），仔细阅读其「Prioritized
  Recommendations」章节，把要接地的方向锁定在这些建议上。
- 如果没有审计报告（Tier1 off），读 docs/plans/SPRINT.md 里所有 [ ] 未完成任务，接地这些任务。

## 工作流程

1. 确定本轮方向（从审计报告的建议 或 SPRINT 的未完成任务）
2. 读 docs/plans/pitfalls.md，了解已知坑
3. **读 docs/project_structure.md（若存在）作为导航地图**，定位本轮方向涉及的目录/文件，避免从零摸整库
4. 针对每个方向，从地图定位的落点出发用 Glob/Grep/Read 探查相关代码：现有架构、关键模块、数据流、相关文件路径
5. 判断可行性与约束：哪些方向受现有设计限制？哪些需要先重构？哪些是"快乐路径"之外没覆盖的？
6. 将结果写入 docs/orch/scout.md（覆盖写），严格分两段：

# Scout Report — Iteration {ITER}

## A. 约束与可行性（给 Planner —— 影响 WHAT/范围）
- [方向标识] 可行性：[直接可做 / 需先重构 / 受限于X无法做 / 范围需收窄为Y]
  - 理由：[基于代码现状的具体原因]
  - 对规划的建议：[这个方向该怎么调整范围/拆分/排序]

## B. 代码地图与坑（给 Generator —— HOW 接地）
- [方向标识]
  - 相关文件：[路径列表，附一句话说明每个文件的角色]
  - 现有架构/数据流：[简述实现这个方向需要理解的现状]
  - 注意的坑：[来自 pitfalls.md 或你探查中发现的，实现时容易踩的]

## C. 新发现的坑（如有，待Sprint结束追加到 pitfalls.md）
- [分类] 描述...
```

描述设为："Scout: 代码接地侦察（第{ITER}轮）"

#### Step C — Planner（Task subagent, general-purpose）

等 Step A/B 完成后，启动 Planner subagent。

根据 `--tier1` / `--mode` / `--scout` 调整 Planner 的输入源和 prompt：

**需求来源（What）：**
- **tier1 on, mode = experience**：读取 `docs/orch/product-audit-report.md`（体验官报告）
- **tier1 on, mode = evolution**：读取 `docs/orch/evolution-audit-report.md`（进化策略报告）
- **tier1 on, mode = research**：读取 `docs/orch/research-audit-report.md`（设计研究报告）
- **tier1 on, mode = all**：同时读取三份报告，综合三个视角制定计划
- **tier1 off**：**没有审计报告**——直接从 `docs/plans/SPRINT.md` 的 `[ ]` 未完成任务规划（无 negotiation.md，无逐条回应步骤）

**接地输入（可行性/约束）：**
- **scout on**：读取 `docs/orch/scout.md` 的「A. 约束与可行性」段，据此调整任务范围/拆分/排序（注意：只读约束段，代码地图段是给 Generator 的，Planner 仍不碰文件路径，保持 WHAT/HOW 分离）
- **scout off**：无 scout.md，照常规划

Prompt 如下（用实际迭代编号替换 `{ITER}`，用实际模式替换 `{MODE}`）：

```
你是 Sprint Planner。你的职责是读取 Reviewer 刚产出的审计报告，结合 Sprint 合同和上轮评估，产出一份精简的任务计划（合同），并对 Reviewer 的建议逐条回应。

## 核心规则

1. 你只负责规划 WHAT 和 WHY，绝不指定 HOW
2. 你不写代码、不指定文件路径、不写实现步骤
3. 你的输出：docs/orch/plan.md（本轮任务计划）+ 更新 docs/plans/SPRINT.md；**仅 Tier1 on（有审计报告）时**额外写 docs/orch/negotiation.md（对 Reviewer 的逐条回应）

## 明确禁止

- 不写代码片段或伪代码
- 不写 "需要读取的文件" / "需要修改的文件"
- 不写 "实现要点" / "实现步骤" / "代码步骤"
- 不写具体的函数名、类名、参数列表
- 不要试图帮 Generator 做技术决策，Generator 是一个资深工程师，他自己会 figure out

## 工作流程

1. **确定需求来源（What）**：
   - 若存在审计报告（docs/orch/product-audit-report.md / evolution-audit-report.md /
     research-audit-report.md 任意一个或多个）：仔细阅读其发现和建议。你可以选择性接受——
     并非所有建议都必须采纳，但要在 negotiation.md 中逐条回应并给出理由。
   - 若没有任何审计报告（Tier1 off）：本轮需求直接来自 docs/plans/SPRINT.md 里的 `[ ]`
     未完成任务。**此时不产出 negotiation.md、不做逐条回应**，只规划如何完成这些既有任务。
2. **读取 Scout 接地报告（若存在 docs/orch/scout.md）**：只读「A. 约束与可行性」段，据此
   调整任务范围/拆分/排序。某方向若 Scout 判为"受限/需先重构"，要在计划里反映出来。
3. 读取 docs/plans/SPRINT.md，了解 Sprint 整体目标
4. **读取 docs/plans/pitfalls.md，筛选与本次任务相关的陷阱**（这是必须步骤）
5. 如果 docs/orch/eval.md 存在（上轮 Evaluator 反馈），仔细分析失败原因，据此调整本轮策略
6. 将本轮要做的任务拆解后**追加**到 docs/plans/SPRINT.md 的任务清单（用 `[ ]` 标记未完成项）

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
   - 来源：[体验官建议 / 进化策略师建议 / 研究员建议 / Planner 自主判断 / 上轮遗留]

## 来自 Reviewer 的改进项（采纳的）
- [建议标题] → 本轮行动：[做什么]

## 相关陷阱（从 pitfalls.md 筛选）
- [分类] 陷阱描述...

## 上轮失败分析（仅迭代 2+ 有 eval.md 时填写）
- 失败原因：...
- 本轮策略调整：...

## 验收命令（从 SPRINT.md 的验收命令章节原样复制）
（粘贴验收命令）

### 输出 3：写入 docs/orch/negotiation.md（覆盖写，**仅 Tier1 on；tier1 off 无审计报告时跳过本输出**）

# Negotiation — Iteration {ITER}

## 对本轮 Reviewer 报告的逐条回应

### 建议 1: [Reviewer 建议标题——从审计报告的「Prioritized Recommendations」章节提取，注明来自体验官、进化策略师还是研究员]
- **决策**：接受 / 拒绝 / 部分接受
- **理由**：[为什么做出这个决策——如果是拒绝，必须写清楚原因；如果是部分接受，说明接受哪些、拒绝哪些]
- **本轮行动**：[如果接受/部分接受，本轮具体会做什么；如果拒绝，写 N/A]

### 建议 2: [同上格式]
...

## 本轮 Planner 自主发现的改进方向（不在 Reviewer 报告中的）
- [改进方向] → 本轮行动：[做什么]
```

描述设为："Planner: 分析 Reviewer 报告，写入 plan.md + negotiation.md + 更新 SPRINT.md"

#### Step D — Generator（Task subagent, general-purpose）

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
3. **若存在 docs/orch/scout.md，读取其「B. 代码地图与坑」段**——相关文件、现有架构、要避开的坑，
   据此直接定位，少走探索弯路（Scout 已经替你接地过了）
4. 按依赖顺序逐个实现任务：
   a. 先看 scout.md 的代码地图（若有）锁定相关文件，再用 Glob/Grep/Read 补充探索现有架构
   b. 实现功能（用 Edit/Write 修改代码）
   c. 每完成一项任务，将 docs/plans/SPRINT.md 中对应的 [ ] 改为 [x]
5. **同步 project_structure.md（防漂移，必须步骤）**：若本轮**新增/移动/删除了文件，或改变了某模块的职责**，
   且存在 docs/project_structure.md，则在同一轮里更新它（落点表、目录职责），使其与改后代码一致；
   若 scout.md 的 B 段指出了既有漂移，一并修正。本轮未动文件结构则无需改动。
6. 全部任务完成后，运行 plan.md 中列出的所有验收命令
7. 将结果写入 docs/orch/gen_status.md，格式如下：

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

## 文件结构变更（防漂移自报）
- 本轮是否新增/移动/删除文件或改变模块职责：是 / 否
- 若是：改动了哪些落点，project_structure.md 是否已同步更新（是/否/项目无此文件）

## 状态
PASSED / PARTIAL / FAILED / BLOCKED

## 注意事项
- 每次修改文件前，必须先读取该文件理解现有代码，不要盲改
- 如果某个任务卡住，记录原因后跳到下一个任务，不要死循环
- 验收命令的输出必须是实际运行结果，不要伪造
```

描述设为："Generator: 实现代码并运行验收命令"

#### Step E — Evaluator（Task subagent, general-purpose）

等 Generator 完成后，启动 Evaluator subagent，prompt 如下（用实际迭代编号替换 `{ITER}`）：

```
你是 QA Evaluator，独立验证 Sprint 是否完成。

## 核心原则：不信任 Generator 的自报结果，自己重新运行所有验收命令。

注意（终止语义跟随 `--tier1`）：
- **tier1 on（默认）**：你的评估报告是给下一轮 Planner 的信息，**不直接终止循环**，引擎跑满全部迭代。
- **tier1 off**：你的决策**直接控制循环**——必须在终端最后输出一行 `DECISION: COMPLETE`（所有 [ ] 已勾选且所有验收命令通过）/ `DECISION: CONTINUE`（还有未完成或失败项）/ `DECISION: BLOCKED`（环境/依赖阻塞），Orchestrator 靠这行决定停或继续（同 multi-ralph）。

## 工作流程

1. 读取 docs/plans/SPRINT.md，逐项检查 checkbox 状态（[x] vs [ ]）
2. **读取 docs/plans/pitfalls.md，确认 Generator 的实现没有踩到已知陷阱**
3. 读取 docs/orch/gen_status.md 了解 Generator 的自报结果
4. 自己重新运行 SPRINT.md 中「验收命令」章节的每一条命令，记录实际输出
5. 对比你的实际结果与 Generator 的自报——是否一致？是否有遗漏？
6. 检查是否有新的陷阱需要记录（Generator 的 gen_status.md 中可能有"新发现的陷阱"）
7. **结构漂移检查（若存在 docs/project_structure.md）**：对照 Generator 实际改动的文件（看 gen_status.md 的「文件结构变更」自报，并自己用 git diff / Glob 核对），确认 project_structure.md 是否与改后代码一致——有没有新增/移动/删除的文件没反映、有没有职责变了没更新。不一致即为漂移，记进报告。**tier1 off 时，未同步的漂移视为本轮未达标，决策应为 CONTINUE。**
8. 将完整评估写入 docs/orch/eval.md，格式如下：

# Evaluator Report — Iteration {ITER}

## Checkbox 状态
（逐项列出每个任务的 [x] / [ ] 状态）

## 验收命令重跑结果
（每条命令及其实际输出——必须是你自己运行的，不是复制 gen_status.md 的）

## Generator 报告 vs 实际对比
（两者是否一致？有哪些出入？）

## pitfalls 合规检查
（Generator 的实现是否违反了 pitfalls.md 中的已知注意事项？）

## 结构漂移检查（若项目有 docs/project_structure.md）
（Generator 改动的文件是否都已反映进 project_structure.md？列出任何未同步项；无漂移则写"一致"）

## 失败原因分析（如有）
（具体什么失败了、可能的修复方向——这些信息会传给下一轮 Planner）

## 新陷阱待追加（如有）
（从 Generator 报告或验证过程中发现的新陷阱，待Sprint结束后追加到 pitfalls.md）

## 决策
COMPLETE / CONTINUE / BLOCKED
（tier1 on 时仅信息性；tier1 off 时必须另在终端最后输出严格一行 `DECISION: COMPLETE` / `DECISION: CONTINUE` / `DECISION: BLOCKED` 供 Orchestrator 解析）

重要：你只能读文件和运行命令来验证，不能修改任何源代码文件。
```

描述设为："Evaluator: 独立验证并输出评估报告"

---

### 迭代完成后的收尾

本轮各阶段完成后，告知用户：

```
第 {ITER}/{MAX_ITER} 轮完成（tier1: {ON/OFF}, mode: {MODE}, scout: {ON/OFF}）
  Reviewer:    [tier1 on 时列出对应报告文件；off 则标注 "skipped (tier1 off)"]
  Scout:       [scout on 时 docs/orch/scout.md；off 则标注 "skipped"]
  Planner:     docs/orch/plan.md（tier1 on 时 +negotiation.md）+ SPRINT.md 更新
  Generator:   docs/orch/gen_status.md
  Evaluator:   docs/orch/eval.md
```

**本轮末尾路由：**
- **tier1 on**：不看 Evaluator 决策，直接进入下一轮（跑满 max_iter）。最后一轮后进入循环结束处理。
- **tier1 off**：解析 Evaluator 终端输出的 `DECISION:` 行（同 multi-ralph）：
  - `COMPLETE` → 向用户报告成功，进入循环结束处理，**提前结束循环**
  - `BLOCKED` → 向用户报告阻塞、展示 eval.md，进入循环结束处理，结束循环
  - `CONTINUE` → 继续下一轮（若已达 max_iter 则进入循环结束处理）

### 循环结束（达到 max_iter，或 tier1 off 时 DECISION 提前 COMPLETE/BLOCKED）

1. 将本轮所有 gen_status.md 和 eval.md 中发现的新陷阱追加到 `docs/plans/pitfalls.md`
2. **tier1 on 时**：将本轮审计报告中新发现的产品问题（如有）也追加到 pitfalls.md
3. 向用户展示最终状态摘要：
   - 各轮 Evaluator 评估结果（tier1 off 时附最终 `DECISION`）
   - **tier1 on 时**额外展示：
     - 最终 Reviewer 综合评分（从最后一轮审计报告提取，experience→product-audit-report.md，evolution→evolution-audit-report.md，research→research-audit-report.md，all→综合三份）
     - Reviewer Top 3 改进建议（从最后一轮审计报告提取）
     - 产品方的最终回应（从最后一轮 negotiation.md 提取）
4. 告知用户完整报告位置

---

## 架构图

```
Orchestrator（当前 Agent 循环；终止跟随 --tier1：on=跑满 max_iter，off=DECISION 控停）
│
├── Step A: Reviewer (系统 agent，**仅 --tier1 on**，根据 --mode 决定) ── Tier 1
│   ├── mode=experience → product-experience-reviewer → 写 product-audit-report.md
│   ├── mode=evolution  → product-evolution-reviewer  → 写 evolution-audit-report.md
│   ├── mode=research   → product-research-reviewer   → 写 research-audit-report.md
│   └── mode=all        → 三者并行启动                → 三份报告
│   （tier1 off：跳过，SPRINT.md 的 [ ] 任务即需求源）
│
├── Step B: Scout     (Task subagent, **仅 --scout on**, 只读) → 接地代码 → 写 scout.md ── Tier 2
├── Step C: Planner   (Task subagent) → 读 审计报告/SPRINT + scout.md约束段 → 写 plan.md (+negotiation.md*) + 追加 SPRINT.md
├── Step D: Generator (Task subagent) → 读 plan.md + scout.md代码地图段 → 实现 + 勾checkbox → 写 gen_status.md
└── Step E: Evaluator (Task subagent) → 读 gen_status.md → 独立复验 → 写 eval.md（tier1 off 时 +DECISION 控停）
                                                                    （* negotiation.md 仅 tier1 on）

共享文件（隔离 subagent 之间的通信介质）：
├── docs/plans/SPRINT.md                    ← Sprint 合同（全部角色可读）
├── docs/plans/pitfalls.md                  ← 陷阱知识库（全部角色必读）
├── docs/project_structure.md               ← 持久代码地图：Scout 读(导航) / Generator 写(防漂移) / Evaluator 守
├── docs/orch/product-audit-report.md       ← Experience Reviewer → Scout/Planner（仅 tier1 on）
├── docs/orch/evolution-audit-report.md     ← Evolution Reviewer → Scout/Planner（仅 tier1 on）
├── docs/orch/research-audit-report.md      ← Research Reviewer → Scout/Planner（仅 tier1 on）
├── docs/orch/scout.md                      ← Scout → Planner(约束段) / Generator(代码地图段)（仅 scout on）
├── docs/orch/negotiation.md                ← Planner → Reviewer（下轮）的协商回应（仅 tier1 on）
├── docs/orch/plan.md                       ← Planner → Generator 的合同
├── docs/orch/gen_status.md                 ← Generator → Evaluator 的交接
└── docs/orch/eval.md                       ← Evaluator → Planner（下轮）的反馈
```

各 subagent 之间无上下文共享，只通过上述文件交互。pitfalls.md 是贯穿所有角色的共享知识。Tier1 on 时 negotiation.md 和审计报告构成"Reviewer → 产品方 → Reviewer"的协商闭环；Scout 则在需求与代码之间架桥（约束段→Planner，代码地图段→Generator）。Tier1 off 时退化为 multi-ralph 式纯执行循环（Scout→P→G→E，DECISION 控停）。
