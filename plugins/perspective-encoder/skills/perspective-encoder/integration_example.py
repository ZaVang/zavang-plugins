"""
Perspective Encoder 与 DeepThought 模型联动示例

展示如何在外生变量推断 -> Mediator 分析 -> BN 推理的完整流程中使用 Perspective Encoder。
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# 导入 perspective-encoder 组件
from scripts.exogenous_inferrer import (
    ExogenousVariableInferrer,
    RegimeState,
    EconCycle,
    MonetaryCycle,
)
from scripts.market_state_tracker import MarketStateTracker
from scripts.perspective_selector import PerspectiveSelector
from scripts.prompt_assembler import PromptAssembler


# ============================================
# DeepThought Schema (简化版，实际从 deepthought.models 导入)
# ============================================

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
class MarketContext:
    """市场上下文，传递给 LLM 进行分析"""
    event_type: EventType
    event_details: Dict[str, Any]
    mediator_snapshot: Dict[str, Dict]
    current_state: Dict[str, Any]
    event_history: List[Dict]


class DeepThoughtPerspectiveEncoder:
    """
    DeepThought 与 Perspective Encoder 的集成类
    
    处理流程：
    1. 接收事件
    2. 判断是否需要更新外生变量
    3. 选择激活的视角
    4. 生成分析 prompt
    5. 返回结构化分析结果
    """
    
    def __init__(self):
        self.inferrer = ExogenousVariableInferrer()
        self.tracker = MarketStateTracker()
        self.selector = PerspectiveSelector()
        self.assembler = PromptAssembler()
    
    def process_event(
        self,
        event_type: EventType,
        event_details: Dict[str, Any],
        mediator_snapshot: Dict[str, Dict],
        event_history: List[Dict],
    ) -> Dict[str, Any]:
        """
        处理事件的完整流程
        
        Args:
            event_type: 事件类型
            event_details: 事件详情
            mediator_snapshot: 当前 mediator 状态快照
            event_history: 近期事件历史
        
        Returns:
            包含状态更新和视角选择的完整分析上下文
        """
        current_state = self.tracker.get_current_state()
        
        # Step 1: 判断是否需要更新外生变量
        state_updates = {}
        
        if self.inferrer.should_infer_regime(event_type.value, current_state):
            regime_prompt = self.inferrer.get_regime_inference_prompt(
                event_type=event_type.value,
                event_details=event_details,
                mediator_snapshot=mediator_snapshot,
                event_history=event_history,
            )
            state_updates["regime_prompt"] = regime_prompt
            # 实际使用时，这里调用 LLM 获取结果
            # regime_result = llm.invoke(regime_prompt)
            # new_regime = self.inferrer.parse_regime_output(regime_result)
            # self.tracker.update_regime(...)
        
        if self.inferrer.should_infer_cycle(event_type.value, current_state):
            cycle_prompt = self.inferrer.get_cycle_inference_prompt(
                event_type=event_type.value,
                event_details=event_details,
                mediator_snapshot=mediator_snapshot,
                macro_indicators=event_details.get("macro", {}),
            )
            state_updates["cycle_prompt"] = cycle_prompt
        
        # Step 2: 选择激活的视角
        active_perspectives = self.selector.select_perspectives(
            regime=current_state.regime,
            event_type=event_type.value,
        )
        
        # Step 3: 生成每个视角的分析 prompt
        analysis_prompts = {}
        for perspective_name in active_perspectives:
            prompt = self.assembler.assemble(
                perspective_name=perspective_name,
                context={
                    "event_type": event_type.value,
                    "event_details": event_details,
                    "mediator_snapshot": mediator_snapshot,
                    "current_state": {
                        "regime": current_state.regime,
                        "econ_cycle": current_state.econ_cycle,
                        "monetary_cycle": current_state.monetary_cycle,
                    },
                },
            )
            analysis_prompts[perspective_name] = prompt
        
        return {
            "current_state": self.tracker.get_summary(),
            "state_updates": state_updates,
            "active_perspectives": active_perspectives,
            "analysis_prompts": analysis_prompts,
        }
    
    def get_mediator_specific_analysis(
        self,
        mediator_name: str,
        mediator_snapshot: Dict[str, Dict],
        current_state: Dict[str, Any],
    ) -> str:
        """
        获取特定 Mediator 的专门分析 prompt
        
        Args:
            mediator_name: Mediator 名称（如 "M_risk_premium"）
            mediator_snapshot: Mediator 状态快照
            current_state: 当前市场状态
        
        Returns:
            该 Mediator 的分析 prompt
        """
        perspective_name = f"{mediator_name}_analyst"
        
        return self.assembler.assemble(
            perspective_name=perspective_name,
            context={
                "mediator_snapshot": mediator_snapshot,
                "current_state": current_state,
            },
        )


# ============================================
# 使用示例
# ============================================

def example_geopolitical_event():
    """地缘政治事件处理示例"""
    encoder = DeepThoughtPerspectiveEncoder()
    
    # 模拟事件
    result = encoder.process_event(
        event_type=EventType.GEOPOLITICAL,
        event_details={
            "description": "地区冲突升级，引发市场担忧",
            "severity": "high",
            "affected_regions": ["中东", "欧洲"],
        },
        mediator_snapshot={
            "M_risk_premium": {
                "direction": "up",
                "magnitude": "strong",
                "confidence": 0.8,
            },
            "M_liquidity": {
                "direction": "down",
                "magnitude": "moderate",
                "confidence": 0.6,
            },
            "M_defensive": {
                "direction": "up",
                "magnitude": "moderate",
                "confidence": 0.7,
            },
        },
        event_history=[
            {"type": "sentiment", "time": "2024-01-15T10:00:00"},
            {"type": "monetary_policy", "time": "2024-01-10T14:00:00"},
        ],
    )
    
    print("=== 处理结果 ===")
    print(f"当前状态: {result['current_state']}")
    print(f"激活视角: {result['active_perspectives']}")
    
    if "regime_prompt" in result["state_updates"]:
        print("\n=== Regime 推断 Prompt (前500字符) ===")
        print(result["state_updates"]["regime_prompt"][:500] + "...")
    
    for name, prompt in result["analysis_prompts"].items():
        print(f"\n=== 视角: {name} (前300字符) ===")
        print(prompt[:300] + "...")


def example_mediator_analysis():
    """Mediator 专门分析示例"""
    encoder = DeepThoughtPerspectiveEncoder()
    
    # 获取 M_risk_premium 的专门分析
    prompt = encoder.get_mediator_specific_analysis(
        mediator_name="M_risk_premium",
        mediator_snapshot={
            "CNTRD_BETA": {"value": 1.2, "percentile": 0.75},
            "CNTRD_LEVERAGE": {"value": 0.45, "percentile": 0.60},
            "CNTRD_EARNVAR": {"value": 0.18, "percentile": 0.30},
        },
        current_state={
            "regime": "risk_off",
            "econ_cycle": "recession",
            "monetary_cycle": "easing",
        },
    )
    
    print("=== M_risk_premium 分析 Prompt ===")
    print(prompt)


if __name__ == "__main__":
    print(">>> 示例 1: 地缘政治事件处理")
    example_geopolitical_event()
    
    print("\n" + "="*60 + "\n")
    
    print(">>> 示例 2: Mediator 专门分析")
    example_mediator_analysis()
