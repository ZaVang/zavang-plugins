# Perspective Encoder Skill

## 概述

Perspective Encoder 是一个模块化的多视角推理框架，用于 deepthought 项目的因子预测系统。

**核心理念**：不同分析视角独立封装，运行时根据市场状态动态组合，生成针对性的 prompt。

---

## 架构设计

```
perspective-encoder/
├── SKILL.md                          # 本文件
├── perspectives/                     # 视角定义（模块化）
│   ├── transmission/                 # 传导路径视角
│   │   ├── policy_transmission.md
│   │   ├── liquidity_analyst.md
│   │   ├── cycle_positioner.md
│   │   ├── risk_appetite_analyst.md
│   │   └── sector_rotator.md
│   ├── regime/                       # Regime 专家视角
│   │   ├── risk_on_specialist.md
│   │   ├── risk_off_specialist.md
│   │   ├── crisis_specialist.md
│   │   └── recovery_specialist.md
│   └── event/                        # 事件类型专家视角
│       ├── monetary_policy_analyst.md
│       ├── geopolitical_analyst.md
│       └── trade_analyst.md
├── scripts/                          # 自动化脚本
│   ├── perspective_selector.py       # 根据状态选择激活视角
│   └── prompt_assembler.py           # 动态拼装 prompt
├── templates/                        # Prompt 模板
│   ├── base_prompt.j2
│   └── mediator_output.j2
└── config/
    └── perspective_registry.yaml     # 视角注册表
```

---

## 使用方式

### 1. 基础用法

```python
from perspective_encoder import PerspectiveEncoder

# 初始化
encoder = PerspectiveEncoder()

# 根据当前状态生成 prompt
context = {
    "regime": "risk_on",
    "econ_cycle": "expansion",
    "monetary_cycle": "neutral",
    "event_type": "monetary_policy",
    "recent_news": ["美联储维持利率不变", "...")
}

# 自动选择视角并拼装 prompt
prompt = encoder.assemble_prompt(context)

# 输出结构化的 mediator 判断
output = encoder.run_perspective(context, perspective_name="policy_transmission")
```

### 2. 手动选择视角

```python
# 只使用流动性视角
prompt = encoder.get_perspective_prompt("liquidity_analyst", context)

# 组合多个视角
prompts = encoder.combine_perspectives(
    ["cycle_positioner", "risk_appetite_analyst"],
    context
)
```

### 3. 获取视角的 mediator 权重

```python
# 查看某视角关注的 mediator
weights = encoder.get_mediator_weights("policy_transmission")
# {"M_rate_expectation": 0.35, "M_rate_sensitive": 0.30, ...}
```

---

## 视角分层

### Layer 0: 外生变量推断视角（Exogenous）

推断市场状态，优先级最高，必须先判断：

| 视角 | 输出 | 触发条件 |
|------|------|---------|
| regime_analyst | RegimeState | 地缘政治/流动性/货币政策/情绪事件 |
| cycle_analyst | EconCycle + MonetaryCycle | 货币政策/财政政策/盈利事件 |

### Layer 1: 传导路径视角（Transmission）

基于因果网络的传导路径设计，每条路径一个视角：

| 视角 | 传导路径 | 关注 Mediator |
|------|---------|--------------|
| policy_transmission | Policy → MonetaryCycle → Rate | M_rate_expectation, M_rate_sensitive |
| liquidity_analyst | Liquidity + MonetaryCycle | M_liquidity |
| cycle_positioner | EconCycle + Regime | M_cyclical, M_defensive |
| risk_appetite_analyst | Regime + Geopolitical/Sentiment | M_risk_premium |
| sector_rotator | EconCycle + Regime + Event | M_sector_rotation, M_growth_tech |

### Layer 2: Regime 专家视角

根据当前 Regime 状态激活：

| 视角 | 触发条件 | 特点 |
|------|---------|------|
| risk_on_specialist | Regime == RISK_ON | 风险偏好扩张期分析 |
| risk_off_specialist | Regime == RISK_OFF | 风险偏好收缩期分析 |
| crisis_specialist | Regime == CRISIS | 危机场景，强调极端场景 |
| recovery_specialist | Regime == RECOVERY | 复苏期分析 |

### Layer 3: 事件专家视角

根据 EventType 激活：

| 视角 | 触发条件 | 特点 |
|------|---------|------|
| monetary_policy_analyst | EventType == MONETARY_POLICY | 货币政策传导分析 |
| geopolitical_analyst | EventType == GEOPOLITICAL | 地缘政治影响分析 |
| trade_analyst | EventType == TRADE | 贸易政策影响分析 |

### Layer 4: Mediator 专门分析视角

每个 Mediator 一个专门分析器，深入分析因子状态：

| 视角 | 目标 Mediator | 关注因子 |
|------|--------------|---------|
| M_risk_premium_analyst | M_risk_premium | BETA, LEVERAGE, EARNVAR |
| M_liquidity_analyst | M_liquidity | LIQUIDTY, SIZE, MIDCAP, RESVOL |
| M_rate_expectation_analyst | M_rate_expectation | BTOP, EARNYILD, DIVYILD, GROWTH |
| M_sector_rotation_analyst | M_sector_rotation | INDMOM, MOMENTUM, STREVRSL, LTREVRSL |
| M_earnings_revision_analyst | M_earnings_revision | PROFIT, EARNQLTY, INVSQLTY, ANLYSTSN |
| M_cyclical_analyst | M_cyclical | ENERGY, CHEMICAL, MTLMIN, ... |
| M_defensive_analyst | M_defensive | BEVTOB, FOODPROD, HEALTH, UTILITY, ... |
| M_rate_sensitive_analyst | M_rate_sensitive | BANKS, DIVFINAN, REALEST |
| M_growth_tech_analyst | M_growth_tech | SOFTWARE, HDWRSE, ELECEQP, ... |
| M_consumer_analyst | M_consumer | APPAREL, RETAIL, CONSDUR, ... |

---

## 输出格式

每个视角输出标准化的 mediator 判断：

```python
{
    "perspective": "policy_transmission",
    "mediator_judgments": {
        "M_rate_expectation": {
            "direction": "up",
            "confidence": 0.75,
            "reasoning": "美联储维持利率不变，市场预期转向降息"
        },
        "M_rate_sensitive": {
            "direction": "up",
            "confidence": 0.70,
            "reasoning": "银行、地产受益于利率预期改善"
        }
    },
    "factor_overrides": {},  # 可选，极端场景时使用
    "historical_analogue": "2019年美联储暂停加息"
}
```

---

## 扩展方式

### 添加新视角

1. 在 `perspectives/` 对应目录下创建 `.md` 文件
2. 在 `config/perspective_registry.yaml` 中注册
3. 脚本会自动加载

### 视角文件格式

```markdown
# 视角名称

## 角色
你是XXX分析师。

## 传导路径
[分析路径描述]

## 关键问题
- 问题1
- 问题2

## Mediator 权重
M_xxx: 0.35
M_yyy: 0.25

## 可用工具
- tool_1
- tool_2

## 历史类比
[历史类似场景]
```

---

## 与 deepthought 的集成

```python
# 在 Pipeline 中使用
from perspective_encoder import PerspectiveEncoder

class LLMReasonerTool:
    def __init__(self):
        self.encoder = PerspectiveEncoder()
    
    async def execute(self, payload: dict) -> dict:
        # 1. 获取当前状态
        context = self._build_context(payload)
        
        # 2. 自动选择视角
        perspectives = self.encoder.select_perspectives(context)
        
        # 3. 并行运行各视角
        results = await asyncio.gather(*[
            self.encoder.run_perspective(context, p)
            for p in perspectives
        ])
        
        # 4. 输出给 BN 量化层
        return self._aggregate_results(results)
```

---

## 理论依据

本 skill 的设计基于 Anthropic 的研究成果：

1. **PSM (Persona Selection Model)**: 角色设定因果影响模型行为
2. **Emotion Concepts**: 情绪/角色可以作为行为"开关"
3. **CoT 忠诚度**: 需要验证模型是否真的采用了预期视角

---

## 核心实现类

### ExogenousVariableInferrer
外生变量推断器，根据事件流推断市场状态：
- `should_infer_regime()`: 判断是否需要重新推断 Regime
- `should_infer_cycle()`: 判断是否需要重新推断周期
- `get_regime_inference_prompt()`: 生成 Regime 推断 prompt
- `get_cycle_inference_prompt()`: 生成周期推断 prompt

### MarketStateTracker
市场状态持久化跟踪器：
- `get_current_state()`: 获取当前市场状态
- `update_regime()`: 更新 Regime 状态
- `update_econ_cycle()`: 更新经济周期
- `update_monetary_cycle()`: 更新货币政策周期
- `get_transition_history()`: 获取状态转换历史
- `get_regime_duration()`: 获取当前 Regime 持续天数

### DeepThoughtPerspectiveEncoder
集成类，提供完整的事件处理流程：
- `process_event()`: 处理事件的完整流程
- `get_mediator_specific_analysis()`: 获取特定 Mediator 的分析 prompt

---

## 文件清单

### 视角定义
- `perspectives/exogenous/`: 2 个外生变量推断视角
- `perspectives/transmission/`: 5 个传导路径视角
- `perspectives/regime/`: 4 个 Regime 专家视角
- `perspectives/event/`: 8 个事件专家视角
- `perspectives/mediator/`: 10 个 Mediator 专门分析视角

### 脚本
- `scripts/perspective_selector.py`: 视角选择逻辑
- `scripts/prompt_assembler.py`: Prompt 拼装逻辑
- `scripts/exogenous_inferrer.py`: 外生变量推断器
- `scripts/market_state_tracker.py`: 市场状态跟踪器

### 配置与模板
- `config/perspective_registry.yaml`: 视角注册表
- `templates/base_prompt.j2`: 基础 prompt 模板

### 示例
- `integration_example.py`: 与 deepthought 模型的集成示例
