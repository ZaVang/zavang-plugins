"""
外生变量推断器 (Exogenous Variable Inferrer)

负责根据事件流推断外生变量状态：
- RegimeState (RISK_ON, RISK_OFF, TRANSITIONAL, CRISIS, RECOVERY)
- EconCycle (EXPANSION, PEAK, RECESSION, TROUGH)
- MonetaryCycle (EASING, NEUTRAL, TIGHTENING)
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import yaml
from pathlib import Path


class RegimeState(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITIONAL = "transitional"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class EconCycle(str, Enum):
    EXPANSION = "expansion"
    PEAK = "peak"
    RECESSION = "recession"
    TROUGH = "trough"


class MonetaryCycle(str, Enum):
    EASING = "easing"
    NEUTRAL = "neutral"
    TIGHTENING = "tightening"


@dataclass
class ExogenousState:
    """外生变量状态容器"""
    regime: RegimeState
    econ_cycle: EconCycle
    monetary_cycle: MonetaryCycle
    regime_confidence: float
    econ_confidence: float
    monetary_confidence: float
    last_updated: str  # ISO timestamp
    trigger_event: str  # 触发状态判断的事件


class ExogenousVariableInferrer:
    """
    外生变量推断器
    
    使用 Perspective Encoder 的视角来推断外生变量状态。
    状态不会频繁变化，只有在重大事件触发时才重新判断。
    """
    
    def __init__(self, registry_path: str = None):
        """初始化推断器"""
        if registry_path is None:
            registry_path = Path(__file__).parent.parent / "config" / "perspective_registry.yaml"
        
        self.registry = self._load_registry(registry_path)
        self.regime_analyst_prompt = self._load_perspective("regime_analyst")
        self.cycle_analyst_prompt = self._load_perspective("cycle_analyst")
    
    def _load_registry(self, path: Path) -> dict:
        """加载视角注册表"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_perspective(self, perspective_name: str) -> str:
        """加载视角 prompt 文件"""
        perspective_config = self.registry.get("exogenous", {}).get(perspective_name, {})
        file_path = perspective_config.get("file", "")
        if not file_path:
            raise ValueError(f"Perspective {perspective_name} not found in registry")
        
        full_path = Path(__file__).parent.parent / file_path
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def should_infer_regime(self, event_type: str, current_state: ExogenousState) -> bool:
        """
        判断是否需要重新推断 Regime
        
        只有特定事件类型才触发 Regime 重判断
        """
        regime_trigger_events = {
            "geopolitical",    # 地缘政治事件
            "liquidity",       # 流动性冲击
            "monetary_policy", # 重大货币政策
            "sentiment",       # 极端情绪事件
        }
        
        return event_type in regime_trigger_events
    
    def should_infer_cycle(self, event_type: str, current_state: ExogenousState) -> bool:
        """
        判断是否需要重新推断周期
        
        周期变化更慢，只有经济相关事件才触发
        """
        cycle_trigger_events = {
            "monetary_policy",
            "fiscal_policy",
            "earnings",  # 大规模盈利预期修正
        }
        
        return event_type in cycle_trigger_events
    
    def get_regime_inference_prompt(
        self,
        event_type: str,
        event_details: dict,
        mediator_snapshot: dict,
        event_history: list,
    ) -> str:
        """
        生成 Regime 推断 prompt
        
        Args:
            event_type: 当前事件类型
            event_details: 事件详情
            mediator_snapshot: 当前 mediator 状态快照
            event_history: 近期事件序列
        
        Returns:
            完整的推断 prompt
        """
        prompt = f"""# 任务：判断当前市场状态 (Regime)

## 视角指导
{self.regime_analyst_prompt}

## 当前信息

### 事件类型
{event_type}

### 事件详情
```yaml
{yaml.dump(event_details, allow_unicode=True)}
```

### Mediator 状态快照
```yaml
{yaml.dump(mediator_snapshot, allow_unicode=True)}
```

### 近期事件序列（最近10个）
```yaml
{yaml.dump(event_history[-10:], allow_unicode=True)}
```

## 输出要求

请根据上述信息和视角指导，输出 YAML 格式的 Regime 判断：

```yaml
regime_analysis:
  current_regime: "risk_on" | "risk_off" | "transitional" | "crisis" | "recovery"
  confidence: 0.0 - 1.0
  
  key_signals:
    - mediator: "M_xxx"
      direction: "up" | "down" | "neutral"
      magnitude: "weak" | "moderate" | "strong"
  
  regime_probability:
    risk_on: 0.0 - 1.0
    risk_off: 0.0 - 1.0
    transitional: 0.0 - 1.0
    crisis: 0.0 - 1.0
    recovery: 0.0 - 1.0
  
  reasoning: "判断理由"
```
"""
        return prompt
    
    def get_cycle_inference_prompt(
        self,
        event_type: str,
        event_details: dict,
        mediator_snapshot: dict,
        macro_indicators: dict,
    ) -> str:
        """
        生成周期推断 prompt
        
        Args:
            event_type: 当前事件类型
            event_details: 事件详情
            mediator_snapshot: 当前 mediator 状态快照
            macro_indicators: 宏观指标数据
        
        Returns:
            完整的推断 prompt
        """
        prompt = f"""# 任务：判断经济周期和货币政策周期

## 视角指导
{self.cycle_analyst_prompt}

## 当前信息

### 事件类型
{event_type}

### 事件详情
```yaml
{yaml.dump(event_details, allow_unicode=True)}
```

### Mediator 状态快照
```yaml
{yaml.dump(mediator_snapshot, allow_unicode=True)}
```

### 宏观指标
```yaml
{yaml.dump(macro_indicators, allow_unicode=True)}
```

## 输出要求

请根据上述信息和视角指导，输出 YAML 格式的周期判断：

```yaml
cycle_analysis:
  econ_cycle:
    current: "expansion" | "peak" | "recession" | "trough"
    confidence: 0.0 - 1.0
    probability:
      expansion: 0.0 - 1.0
      peak: 0.0 - 1.0
      recession: 0.0 - 1.0
      trough: 0.0 - 1.0
  
  monetary_cycle:
    current: "easing" | "neutral" | "tightening"
    confidence: 0.0 - 1.0
  
  reasoning: "判断理由"
```
"""
        return prompt
    
    def parse_regime_output(self, output_yaml: str) -> dict:
        """解析 Regime 推断输出"""
        try:
            result = yaml.safe_load(output_yaml)
            return result.get("regime_analysis", {})
        except yaml.YAMLError:
            return {}
    
    def parse_cycle_output(self, output_yaml: str) -> dict:
        """解析周期推断输出"""
        try:
            result = yaml.safe_load(output_yaml)
            return result.get("cycle_analysis", {})
        except yaml.YAMLError:
            return {}


# 使用示例
if __name__ == "__main__":
    inferrer = ExogenousVariableInferrer()
    
    # 生成 Regime 推断 prompt
    prompt = inferrer.get_regime_inference_prompt(
        event_type="geopolitical",
        event_details={"description": "地区冲突升级", "severity": "high"},
        mediator_snapshot={
            "M_risk_premium": {"direction": "up", "magnitude": "strong"},
            "M_liquidity": {"direction": "down", "magnitude": "moderate"},
        },
        event_history=[
            {"type": "sentiment", "time": "2024-01-15"},
            {"type": "monetary_policy", "time": "2024-01-10"},
        ],
    )
    
    print(prompt)
