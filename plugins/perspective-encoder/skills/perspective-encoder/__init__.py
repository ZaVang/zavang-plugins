"""
Perspective Encoder - 多视角推理框架

用于 deepthought 项目的因子预测系统。
"""

from pathlib import Path
from typing import Optional
import yaml

from .scripts.perspective_selector import PerspectiveSelector, PerspectiveInfo
from .scripts.prompt_assembler import PromptAssembler


class PerspectiveEncoder:
    """
    Perspective Encoder 统一入口
    
    使用示例：
        encoder = PerspectiveEncoder()
        
        context = {
            "regime": "risk_on",
            "econ_cycle": "expansion",
            "monetary_cycle": "neutral",
            "event_type": "monetary_policy",
            "recent_news": ["美联储维持利率不变"]
        }
        
        # 自动选择视角并生成 prompt
        prompts = encoder.encode(context)
    """
    
    def __init__(self, skill_root: str = None):
        if skill_root is None:
            skill_root = Path(__file__).parent
        
        self.skill_root = Path(skill_root)
        self.selector = PerspectiveSelector(
            self.skill_root / "config" / "perspective_registry.yaml"
        )
        self.assembler = PromptAssembler(str(self.skill_root))
        
        # 加载注册表
        with open(self.skill_root / "config" / "perspective_registry.yaml", 'r', encoding='utf-8') as f:
            self.registry = yaml.safe_load(f)
    
    def encode(
        self,
        context: dict,
        strategy: str = "default"
    ) -> list[tuple[str, str]]:
        """
        根据上下文自动选择视角并生成 prompt
        
        Args:
            context: 市场状态上下文
            strategy: 组合策略 ("default", "minimal", "crisis_mode")
        
        Returns:
            [(perspective_key, prompt), ...]
        """
        # 选择视角
        perspective_keys = self.selector.select(
            regime=context.get('regime', 'unknown'),
            event_type=context.get('event_type', 'none'),
            strategy=strategy
        )
        
        # 拼装 prompt
        return self.assembler.assemble_combined(
            perspective_keys,
            context,
            self.registry
        )
    
    def get_perspective_prompt(
        self,
        perspective_key: str,
        context: dict
    ) -> str:
        """获取单个视角的 prompt"""
        # 获取配置
        config = None
        for category in ['transmission', 'regime', 'event']:
            if perspective_key in self.registry.get(category, {}):
                config = self.registry[category][perspective_key]
                break
        
        # 加载视角文件
        md_content = self.assembler._load_perspective_md(perspective_key, self.registry)
        
        if md_content:
            return self.assembler.assemble(perspective_key, md_content, context, config)
        return ""
    
    def select_perspectives(
        self,
        regime: str,
        event_type: str = "none",
        strategy: str = "default"
    ) -> list[str]:
        """选择激活的视角"""
        return self.selector.select(regime, event_type, strategy)
    
    def get_mediator_weights(self, perspective_keys: list[str]) -> dict:
        """获取多视角的 mediator 权重融合"""
        return self.selector.get_mediator_weights(perspective_keys)
    
    def get_perspective_info(self, perspective_key: str) -> Optional[PerspectiveInfo]:
        """获取视角详细信息"""
        return self.selector.get_perspective_info(perspective_key)
    
    def list_all_perspectives(self) -> dict:
        """列出所有可用视角"""
        result = {
            'transmission': [],
            'regime': [],
            'event': []
        }
        for category in ['transmission', 'regime', 'event']:
            for key, config in self.registry.get(category, {}).items():
                result[category].append({
                    'key': key,
                    'name': config['name']
                })
        return result


# 便捷函数
def encode_perspectives(context: dict, strategy: str = "default") -> list[tuple[str, str]]:
    """便捷函数：编码视角"""
    encoder = PerspectiveEncoder()
    return encoder.encode(context, strategy)


def get_prompt(perspective_key: str, context: dict) -> str:
    """便捷函数：获取单个视角 prompt"""
    encoder = PerspectiveEncoder()
    return encoder.get_perspective_prompt(perspective_key, context)
