# Python 开发规范

## Schema First（Pydantic）
- 所有模块间传递的数据对象必须使用 Pydantic `BaseModel` 定义
- 序列化统一用 `model.model_dump(mode="json")`，不用 `.dict()`
- 字段必须有类型注解；可选字段用 `Optional[T] = None`
- 输出Schema只增不改：后续迭代新增字段，不修改已有字段的类型或含义

## 模块解耦
- 模块间通过 Pydantic Schema 传参，禁止传递 dict 或裸字典作为 API 边界
- 跨层调用只通过接口类（ABC）或 Schema，不依赖具体实现类
- 每个组件（DataFetcher/NewsIngester/Preprocessor/Sentinel/Reasoner/Logger）是独立可替换单元

## 文件组织
- 每个模块的公共接口放在 `__init__.py` 中显式 export
- 测试文件与被测模块一一对应

## 错误处理
- 只在系统边界（CLI 入口、外部 API 调用）做 try/except
- 内部函数抛出具体异常类型，不吞异常

## 异步
- LLM Gateway 是 async 的
- Pipeline 的运行方法需为 `async def`
- CLI 用 `asyncio.run()` 驱动

## 日志
- 所有系统行为通过 EventLogger 记录为追加写入事件流
- 禁止在提交的代码中保留 `print()` 调试语句（用 `logging`）
- LLM 调用必须记录完整的 prompt 和 response

## 禁止事项
- 禁止在业务代码中直接 import 任何 LLM Provider SDK（openai / anthropic / google）
- 禁止硬编码任何 API Key
- 禁止在事件日志中省略任何步骤的输入输出