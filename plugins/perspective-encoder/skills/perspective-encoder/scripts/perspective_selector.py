"""
Perspective Selector - 根据市场状态选择激活的视角

职责：
1. 加载视角注册表
2. 根据当前 Regime/EventType 选择视角
3. 按优先级排序
"""

import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class RegimeState(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITIONAL = "transitional"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class EventType(str, Enum):
    NONE = "none"
    MONETARY_POLICY = "monetary_policy"
    FISCAL_POLICY = "fiscal_policy"
    GEOPOLITICAL = "geopolitical"
    EARNINGS = "earnings"
    TRADE = "trade"
    REGULATORY = "regulatory"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"


@dataclass
class PerspectiveInfo:
    """视角信息"""
    name: str
    file: str
    priority: int
    trigger_type: str  # "always", "regime", "event"
    trigger_value: Optional[str] = None
    mediator_weights: Optional[dict] = None
    mediator_bias: Optional[dict] = None


class PerspectiveSelector:
    """视角选择器"""
    
    def __init__(self, registry_path: str = None):
        if registry_path is None:
            registry_path = Path(__file__).parent.parent / "config" / "perspective_registry.yaml"
        
        with open(registry_path, 'r', encoding='utf-8') as f:
            self.registry = yaml.safe_load(f)
        
        self._build_index()
    
    def _build_index(self):
        """构建视角索引"""
        self.perspectives = {}
        self.regime_map = {}
        self.event_map = {}
        
        for category in ['transmission', 'regime', 'event']:
            for key, config in self.registry.get(category, {}).items():
                info = PerspectiveInfo(
                    name=config['name'],
                    file=config['file'],
                    priority=config.get('priority', 1),
                    trigger_type=category,
                )
                
                # 解析触发条件
                trigger = config.get('trigger', 'always')
                if isinstance(trigger, dict):
                    if 'regime' in trigger:
                        info.trigger_type = 'regime'
                        info.trigger_value = trigger['regime']
                        self.regime_map[trigger['regime']] = key
                    elif 'event_type' in trigger:
                        info.trigger_type = 'event'
                        info.trigger_value = trigger['event_type']
                        self.event_map[trigger['event_type']] = key
                
                # 存储 mediator 相关配置
                if 'mediator_weights' in config:
                    info.mediator_weights = config['mediator_weights']
                if 'mediator_bias' in config:
                    info.mediator_bias = config['mediator_bias']
                
                self.perspectives[key] = info
    
    def select(
        self,
        regime: str,
        event_type: str = "none",
        strategy: str = "default"
    ) -> list[str]:
        """
        选择激活的视角
        
        Args:
            regime: 当前市场 regime
            event_type: 当前事件类型
            strategy: 组合策略 ("default", "minimal", "crisis_mode")
        
        Returns:
            激活的视角 key 列表，按优先级排序
        """
        selected = []
        
        if strategy == "minimal":
            # 最小模式：周期 + 风险偏好 + regime 专家
            selected = ["cycle_positioner", "risk_appetite_analyst"]
            if regime in self.regime_map:
                selected.append(self.regime_map[regime])
        
        elif strategy == "crisis_mode" or regime == "crisis":
            # 危机模式：危机专家优先
            selected = ["crisis_specialist", "liquidity_analyst", "risk_appetite_analyst"]
        
        else:  # default
            # 默认模式：所有传导视角 + regime 专家 + event 专家
            # 传导视角
            for key, info in self.perspectives.items():
                if info.trigger_type == 'transmission':
                    selected.append(key)
            
            # Regime 专家
            if regime in self.regime_map:
                selected.append(self.regime_map[regime])
            
            # Event 专家
            if event_type != "none" and event_type in self.event_map:
                selected.append(self.event_map[event_type])
        
        # 按优先级排序
        selected.sort(key=lambda k: self.perspectives[k].priority, reverse=True)
        
        return selected
    
    def get_perspective_info(self, key: str) -> PerspectiveInfo:
        """获取视角详细信息"""
        return self.perspectives.get(key)
    
    def get_mediator_weights(self, perspective_keys: list[str]) -> dict:
        """
        获取多个视角的 mediator 权重融合
        
        Args:
            perspective_keys: 视角 key 列表
        
        Returns:
            融合后的 mediator 权重
        """
        total_weight = {}
        count = {}
        
        for key in perspective_keys:
            info = self.perspectives.get(key)
            if info and info.mediator_weights:
                for mediator, weight in info.mediator_weights.items():
                    total_weight[mediator] = total_weight.get(mediator, 0) + weight
                    count[mediator] = count.get(mediator, 0) + 1
        
        # 平均化
        return {m: total_weight[m] / count[m] for m in total_weight}
    
    def get_mediator_bias(self, perspective_keys: list[str]) -> dict:
        """
        获取视角的 mediator 方向偏向
        
        Returns:
            mediator -> direction 偏向
        """
        biases = {}
        for key in perspective_keys:
            info = self.perspectives.get(key)
            if info and info.mediator_bias:
                for mediator, direction in info.mediator_bias.items():
                    biases[mediator] = direction  # 后面的覆盖前面的
        return biases


# 使用示例
if __name__ == "__main__":
    selector = PerspectiveSelector()
    
    # 示例 1：正常市场
    perspectives = selector.select(regime="risk_on", event_type="monetary_policy")
    print("Risk on + Monetary policy:", perspectives)
    
    # 示例 2：危机模式
    perspectives = selector.select(regime="crisis")
    print("Crisis mode:", perspectives)
    
    # 示例 3：获取 mediator 权重
    weights = selector.get_mediator_weights(["policy_transmission", "liquidity_analyst"])
    print("Merged weights:", weights)
