# {{PROJECT_NAME}}

## 任务规划原则

1. **Think Deeper First** - 开始任务前充分思考，复杂任务必须先分解为 3+ 个步骤
2. **Implementation Plan** - 复杂任务写入 `docs/implementation_plan.md`，与用户确认后再执行
3. **渐进式开发** - 遵循最小改动原则，代码越精简越好

## 代码实现原则

### 1. 架构优先
- **先搭框架**: 在实现具体功能前，先设计好整体架构和模块划分
- **模块解耦**: 功能模块分离到独立文件，不要堆积在一个文件里
- **单一职责**: 每个模块/文件只负责一个功能域

### 2. 骨架先行
- **先写骨架**: 先实现函数签名和类结构，细节用 TODO 或 docstring 占位
- **用户确认**: 骨架完成后等用户确认再实现具体逻辑
- **示例**:
  ```python
  def process_data(data: InputSchema) -> OutputSchema:
      """处理输入数据并返回结果。

      TODO: 实现数据处理逻辑
      """
      raise NotImplementedError
  ```

### 3. Schema 优先（Python）
- 实现功能前先用 Pydantic 定义数据模型
- Schema 统一放在 `src/schemas/` 目录
- 使用 Pydantic 类而非 dataclass

## 项目结构

```
{{PROJECT_STRUCTURE}}
```

## 快速命令

```bash
# 运行测试
{{TEST_COMMAND}}

# 启动开发
{{DEV_COMMAND}}
```
