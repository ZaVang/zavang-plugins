"""
市场状态跟踪器 (Market State Tracker)

负责持久化跟踪市场状态，包括：
- RegimeState
- EconCycle  
- MonetaryCycle

状态变化时会记录状态转换历史。
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from enum import Enum


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
class StateTransition:
    """状态转换记录"""
    timestamp: str
    state_type: str  # "regime", "econ_cycle", "monetary_cycle"
    from_state: str
    to_state: str
    trigger_event: str
    confidence: float


@dataclass  
class MarketState:
    """市场状态快照"""
    regime: str
    regime_confidence: float
    regime_updated_at: str
    
    econ_cycle: str
    econ_confidence: float
    econ_updated_at: str
    
    monetary_cycle: str
    monetary_confidence: float
    monetary_updated_at: str
    
    last_event_type: str
    last_event_time: str


class MarketStateTracker:
    """
    市场状态跟踪器
    
    状态持久化，记录状态转换历史。
    状态不会频繁变化，只有重大事件才触发状态判断。
    """
    
    def __init__(self, storage_path: str = None):
        """
        初始化跟踪器
        
        Args:
            storage_path: 状态存储路径，默认为 ./market_state/
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "market_state"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.storage_path / "current_state.json"
        self.history_file = self.storage_path / "transition_history.jsonl"
        
        # 加载当前状态
        self._load_state()
    
    def _load_state(self) -> MarketState:
        """加载当前状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return MarketState(**data)
        else:
            # 默认初始状态
            default_state = MarketState(
                regime=RegimeState.TRANSITIONAL.value,
                regime_confidence=0.5,
                regime_updated_at=datetime.now().isoformat(),
                econ_cycle=EconCycle.EXPANSION.value,
                econ_confidence=0.5,
                econ_updated_at=datetime.now().isoformat(),
                monetary_cycle=MonetaryCycle.NEUTRAL.value,
                monetary_confidence=0.5,
                monetary_updated_at=datetime.now().isoformat(),
                last_event_type="none",
                last_event_time=datetime.now().isoformat(),
            )
            self._save_state(default_state)
            return default_state
    
    def _save_state(self, state: MarketState):
        """保存当前状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, indent=2, ensure_ascii=False)
    
    def _record_transition(self, transition: StateTransition):
        """记录状态转换历史"""
        with open(self.history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(transition), ensure_ascii=False) + "\n")
    
    def get_current_state(self) -> MarketState:
        """获取当前市场状态"""
        return self._load_state()
    
    def update_regime(
        self,
        new_regime: RegimeState,
        confidence: float,
        trigger_event: str,
    ) -> bool:
        """
        更新 Regime 状态
        
        Args:
            new_regime: 新的 Regime
            confidence: 置信度
            trigger_event: 触发事件描述
        
        Returns:
            是否发生状态变化
        """
        current = self._load_state()
        
        if current.regime != new_regime.value:
            # 记录状态转换
            transition = StateTransition(
                timestamp=datetime.now().isoformat(),
                state_type="regime",
                from_state=current.regime,
                to_state=new_regime.value,
                trigger_event=trigger_event,
                confidence=confidence,
            )
            self._record_transition(transition)
            
            # 更新状态
            current.regime = new_regime.value
            current.regime_confidence = confidence
            current.regime_updated_at = datetime.now().isoformat()
            current.last_event_type = trigger_event
            current.last_event_time = datetime.now().isoformat()
            
            self._save_state(current)
            return True
        
        return False
    
    def update_econ_cycle(
        self,
        new_cycle: EconCycle,
        confidence: float,
        trigger_event: str,
    ) -> bool:
        """
        更新经济周期状态
        
        Args:
            new_cycle: 新的经济周期
            confidence: 置信度
            trigger_event: 触发事件描述
        
        Returns:
            是否发生状态变化
        """
        current = self._load_state()
        
        if current.econ_cycle != new_cycle.value:
            # 记录状态转换
            transition = StateTransition(
                timestamp=datetime.now().isoformat(),
                state_type="econ_cycle",
                from_state=current.econ_cycle,
                to_state=new_cycle.value,
                trigger_event=trigger_event,
                confidence=confidence,
            )
            self._record_transition(transition)
            
            # 更新状态
            current.econ_cycle = new_cycle.value
            current.econ_confidence = confidence
            current.econ_updated_at = datetime.now().isoformat()
            
            self._save_state(current)
            return True
        
        return False
    
    def update_monetary_cycle(
        self,
        new_cycle: MonetaryCycle,
        confidence: float,
        trigger_event: str,
    ) -> bool:
        """
        更新货币政策周期状态
        
        Args:
            new_cycle: 新的货币政策周期
            confidence: 置信度
            trigger_event: 触发事件描述
        
        Returns:
            是否发生状态变化
        """
        current = self._load_state()
        
        if current.monetary_cycle != new_cycle.value:
            # 记录状态转换
            transition = StateTransition(
                timestamp=datetime.now().isoformat(),
                state_type="monetary_cycle",
                from_state=current.monetary_cycle,
                to_state=new_cycle.value,
                trigger_event=trigger_event,
                confidence=confidence,
            )
            self._record_transition(transition)
            
            # 更新状态
            current.monetary_cycle = new_cycle.value
            current.monetary_confidence = confidence
            current.monetary_updated_at = datetime.now().isoformat()
            
            self._save_state(current)
            return True
        
        return False
    
    def get_transition_history(
        self,
        state_type: str = None,
        limit: int = 50,
    ) -> List[StateTransition]:
        """
        获取状态转换历史
        
        Args:
            state_type: 筛选状态类型，None 表示全部
            limit: 返回记录数量限制
        
        Returns:
            状态转换历史列表
        """
        if not self.history_file.exists():
            return []
        
        transitions = []
        with open(self.history_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                transition = StateTransition(**data)
                
                if state_type is None or transition.state_type == state_type:
                    transitions.append(transition)
        
        return transitions[-limit:]
    
    def get_regime_duration(self) -> int:
        """
        获取当前 Regime 持续天数
        
        Returns:
            持续天数
        """
        current = self._load_state()
        updated = datetime.fromisoformat(current.regime_updated_at)
        return (datetime.now() - updated).days
    
    def get_summary(self) -> dict:
        """获取状态摘要"""
        current = self._load_state()
        return {
            "regime": {
                "state": current.regime,
                "confidence": current.regime_confidence,
                "duration_days": self.get_regime_duration(),
            },
            "econ_cycle": {
                "state": current.econ_cycle,
                "confidence": current.econ_confidence,
            },
            "monetary_cycle": {
                "state": current.monetary_cycle,
                "confidence": current.monetary_confidence,
            },
            "last_event": {
                "type": current.last_event_type,
                "time": current.last_event_time,
            },
        }


# 使用示例
if __name__ == "__main__":
    tracker = MarketStateTracker()
    
    # 获取当前状态
    print("当前状态:", tracker.get_summary())
    
    # 更新 Regime
    changed = tracker.update_regime(
        new_regime=RegimeState.RISK_OFF,
        confidence=0.75,
        trigger_event="geopolitical: 地区冲突升级",
    )
    print(f"Regime 变化: {changed}")
    
    # 查看转换历史
    history = tracker.get_transition_history(state_type="regime", limit=5)
    for t in history:
        print(f"  {t.timestamp}: {t.from_state} -> {t.to_state}")
