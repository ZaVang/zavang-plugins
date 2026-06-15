"""harnesslib.tools — 通用Tool基类。

- LLMTool: "渲染模板 -> 调 LLM -> 返回结果" 的通用模式。
- DataTool: "调适配器取数据" 的通用模式。
"""

from .llm_tool import LLMTool, set_log_prompts
from .data_tool import DataTool

__all__ = ["LLMTool", "DataTool", "set_log_prompts"]
