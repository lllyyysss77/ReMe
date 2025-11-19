# Context Offload Operations

This module provides operations to reduce token usage in conversation contexts.

## Operations

### 1. ContextCompactOp (简单截断)
- **功能**: 通过截断和文件存储来压缩大型工具消息
- **方式**: 保留消息的前 N 个字符，将完整内容保存到文件
- **适用场景**: 工具调用结果非常大但不需要完整内容的情况

### 2. ContextCompressOp (LLM 智能压缩)
- **功能**: 使用语言模型智能压缩对话历史
- **方式**: LLM 生成对话摘要，保留关键信息
- **适用场景**: 需要保持对话连贯性和上下文理解的情况
- **可溯源**: 原始消息保存到文件，可以随时查看完整内容

## ContextCompressOp 使用说明

### 基本用法

```python
from reme_ai.context.offload.context_compress_op import ContextCompressOp

# 创建压缩操作
compress_op = ContextCompressOp(
    all_token_threshold=10000,      # Token 阈值
    keep_recent=5,                  # 保留最近 5 条消息
    compress_system_message=False,  # 不压缩系统消息
    storage_path="./compressed_contexts"  # 原始消息存储路径
)

# 执行压缩
await compress_op.async_call(messages=your_messages)

# 获取结果
compressed_messages = compress_op.context.response.answer
original_path = compress_op.context.response.metadata["original_messages_path"]
```

### 可溯源特性

当消息被压缩后，原始消息会被保存到 JSON 文件中，你可以：

1. **在压缩消息中看到文件路径**:
```
[Compressed conversation history]
[LLM 生成的摘要内容...]

(Original 6 messages are stored in: ./compressed_contexts/context_20241118_143520_123456.json)
```

2. **从 metadata 中获取路径**:
```python
original_path = compress_op.context.response.metadata["original_messages_path"]
compressed_count = compress_op.context.response.metadata["compressed_message_count"]
```

3. **读取原始消息**:
```python
import json

with open(original_path, 'r', encoding='utf-8') as f:
    original_messages = json.load(f)

# original_messages 是一个列表，包含所有原始消息
# 每条消息包含: role, content, name, tool_call_id 等字段
```

### 文件格式

保存的 JSON 文件格式示例：

```json
[
  {
    "role": "user",
    "content": "I need help building a REST API...",
    "name": null,
    "tool_call_id": null
  },
  {
    "role": "assistant",
    "content": "Great choice! FastAPI is...",
    "name": null,
    "tool_call_id": null
  }
]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `all_token_threshold` | int | 20000 | 触发压缩的 token 阈值 |
| `keep_recent` | int | 5 | 保留最近多少条消息不压缩 |
| `compress_system_message` | bool | False | 是否压缩系统消息 |
| `storage_path` | str | "./compressed_contexts" | 原始消息存储目录 |

### 压缩流程

1. **分离消息**: 
   - 系统消息（可选择是否压缩，默认不压缩）
   - 待压缩消息（较旧的消息，排除最近的 N 条）
   - 最近消息（保持原样）
2. **检查 token 数**: 只计算待压缩消息的 token 数，如果低于阈值，不压缩
3. **保存原始消息**: 将待压缩的消息保存到 JSON 文件
4. **LLM 压缩**: 调用语言模型生成简洁摘要
5. **构建新上下文**: 系统消息 + 压缩摘要 + 最近消息
6. **添加溯源信息**: 在压缩消息中添加原始文件路径

> **重要**: Token 阈值只针对待压缩的消息部分（不包括系统消息和最近消息），这样可以更精确地控制压缩时机。

### 测试示例

运行测试脚本查看完整示例：

```bash
python reme_ai/context/offload/test_context_compress.py
```

测试包括：
- Test 1: Token 数低于阈值（不压缩）
- Test 2: Token 数超过阈值（执行压缩）
- Test 3: 包含系统消息的压缩
- Test 4: 演示可溯源性（读取原始文件）

### 优势

✅ **智能压缩**: 使用 LLM 理解内容，生成高质量摘要  
✅ **保留上下文**: 保持对话连贯性和关键信息  
✅ **完全可溯源**: 所有原始内容都保存在文件中  
✅ **灵活配置**: 可以控制压缩阈值、保留消息数等  
✅ **透明可见**: 压缩后的消息明确标注原始文件位置  
✅ **精确控制**: Token 阈值只针对待压缩内容，避免因系统消息触发不必要的压缩

### 逻辑优势示例

假设有以下对话：
- 1 条系统消息 (2000 tokens)
- 10 条历史消息 (8000 tokens)
- 3 条最近消息 (3000 tokens)

**旧逻辑**（先计算总 token）:
```
总 token = 13000 > 阈值 10000 → 触发压缩
```

**新逻辑**（先分离，只计算待压缩部分）:
```
系统消息 (2000 tokens) - 保留
待压缩消息 (8000 tokens) - 检查：8000 < 阈值 10000 → 不压缩
最近消息 (3000 tokens) - 保留
```

这样可以避免因为系统消息占用大量 token 而触发不必要的压缩！  

### 注意事项

⚠️ **存储空间**: 原始消息会保存到磁盘，注意定期清理旧文件  
⚠️ **LLM 成本**: 每次压缩都会调用 LLM，会产生 API 成本  
⚠️ **压缩质量**: 压缩质量取决于使用的 LLM 模型能力  
⚠️ **文件管理**: 需要自行管理存储目录，避免文件过多  

