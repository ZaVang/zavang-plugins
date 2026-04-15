"""
Prompt Assembler - 动态拼装视角 prompt

职责：
1. 加载视角模板
2. 填充上下文信息
3. 生成最终 prompt
"""

import yaml
from pathlib import Path
from jinja2 import Template
from typing import Optional


class PromptAssembler:
    """Prompt 拼装器"""
    
    def __init__(self, skill_root: str = None):
        if skill_root is None:
            skill_root = Path(__file__).parent.parent
        
        self.skill_root = Path(skill_root)
        self.perspectives_dir = self.skill_root / "perspectives"
        self.templates_dir = self.skill_root / "templates"
        
        # 加载基础模板
        self.base_template = self._load_template("base_prompt.j2")
    
    def _load_template(self, template_name: str) -> Template:
        """加载 Jinja2 模板"""
        template_path = self.templates_dir / template_name
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return Template(f.read())
        return None
    
    def _load_perspective_md(self, perspective_key: str, registry: dict) -> str:
        """加载视角 markdown 文件"""
        # 从注册表找到文件路径
        for category in ['transmission', 'regime', 'event']:
            if perspective_key in registry.get(category, {}):
                file_path = registry[category][perspective_key]['file']
                full_path = self.skill_root / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        return f.read()
        return ""
    
    def assemble(
        self,
        perspective_key: str,
        perspective_md: str,
        context: dict,
        registry_info: dict = None
    ) -> str:
        """
        拼装单个视角的 prompt
        
        Args:
            perspective_key: 视角 key
            perspective_md: 视角 markdown 内容
            context: 上下文信息
            registry_info: 注册表中的视角配置
        
        Returns:
            完整的 prompt
        """
        # 提取视角中的关键部分
        role_section = self._extract_section(perspective_md, "角色")
        path_section = self._extract_section(perspective_md, "传导路径")
        questions_section = self._extract_section(perspective_md, "关键问题")
        framework_section = self._extract_section(perspective_md, "判断框架")
        
        # 获取 mediator 权重
        mediator_weights = {}
        if registry_info and 'mediator_weights' in registry_info:
            mediator_weights = registry_info['mediator_weights']
        
        # 构建 prompt
        prompt = f"""# 角色设定

{role_section}

---

# 当前市场状态

- **Regime**: {context.get('regime', 'unknown')}
- **EconCycle**: {context.get('econ_cycle', 'unknown')}
- **MonetaryCycle**: {context.get('monetary_cycle', 'unknown')}
- **EventType**: {context.get('event_type', 'none')}
- **近期事件**: {context.get('recent_news', ['无'])[0] if context.get('recent_news') else '无'}

---

# 分析框架

## 传导路径

{path_section}

## 关键问题

{questions_section}

---

# 输出要求

请对以下 mediator 给出方向判断和置信度：

"""
        
        # 添加 mediator 列表
        if mediator_weights:
            prompt += "**重点关注（权重从高到低）**:\n"
            sorted_mediators = sorted(mediator_weights.items(), key=lambda x: x[1], reverse=True)
            for mediator, weight in sorted_mediators:
                prompt += f"- {mediator} (权重: {weight:.0%})\n"
        else:
            prompt += "请判断所有 10 个 mediator 的方向。\n"
        
        prompt += """
## 输出格式

```json
{
  "perspective": "视角名称",
  "mediator_judgments": {
    "M_xxx": {
      "direction": "up/down/stable",
      "confidence": 0.0-1.0,
      "reasoning": "判断依据"
    }
  },
  "historical_analogue": "历史类比场景",
  "factor_overrides": {}  // 可选，极端场景时使用
}
```

---

# 可用工具

你可以调用以下工具获取信息：
- query_factor_history: 查询历史因子表现
- query_bn_prediction: 查询 BN 基准预测
- query_recent_returns: 查询近期因子走势
- query_constraints: 查询因子约束

---

请开始你的分析。
"""
        
        return prompt
    
    def _extract_section(self, md_content: str, section_title: str) -> str:
        """从 markdown 中提取指定章节"""
        lines = md_content.split('\n')
        in_section = False
        section_lines = []
        
        for line in lines:
            if line.startswith('## ') and section_title in line:
                in_section = True
                continue
            elif line.startswith('## ') and in_section:
                break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
    
    def assemble_combined(
        self,
        perspective_keys: list[str],
        context: dict,
        registry: dict
    ) -> list[tuple[str, str]]:
        """
        拼装多个视角的 prompt
        
        Returns:
            [(perspective_key, prompt), ...]
        """
        results = []
        
        for key in perspective_keys:
            # 找到配置
            config = None
            for category in ['transmission', 'regime', 'event']:
                if key in registry.get(category, {}):
                    config = registry[category][key]
                    break
            
            # 加载视角文件
            md_content = self._load_perspective_md(key, registry)
            
            if md_content:
                prompt = self.assemble(key, md_content, context, config)
                results.append((key, prompt))
        
        return results


# 使用示例
if __name__ == "__main__":
    assembler = PromptAssembler()
    
    # 示例上下文
    context = {
        "regime": "risk_on",
        "econ_cycle": "expansion",
        "monetary_cycle": "neutral",
        "event_type": "monetary_policy",
        "recent_news": ["美联储维持利率不变，声明偏鸽"]
    }
    
    # 加载注册表
    import yaml
    registry_path = Path(__file__).parent.parent / "config" / "perspective_registry.yaml"
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = yaml.safe_load(f)
    
    # 拼装单个视角
    md_content = assembler._load_perspective_md("policy_transmission", registry)
    prompt = assembler.assemble("policy_transmission", md_content, context)
    print("=== Policy Transmission Prompt ===")
    print(prompt[:500])
