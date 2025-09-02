# ReMe (formerly memoryscope)

<p align="center">
 <img src="doc/figure/reme_logo.jpg" alt="ReMe.ai Logo" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python Version"></a>
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/pypi-v1.0.0-blue?logo=pypi" alt="PyPI Version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="https://github.com/modelscope/ReMe"><img src="https://img.shields.io/github/stars/modelscope/ReMe?style=social" alt="GitHub Stars"></a>
</p>

<p align="center">
  <strong>ReMe: 为agent设计的记忆管理框架</strong><br>
  <em>Remember Me, Refine Me</em>
</p>

---

agent时代的记忆不单是用于保存个性化的用户信息。agent在完成任务时，我们希望它能够有区分性地记住用户的偏好，以及如何正确地行动。 

当智能体处理任务时，它都在重新发明无数其他智能体已经发现的解决方案。这就像要求每个人都从头发现火、农业和数学一样。

ReMe为AI智能体提供了统一的记忆与经验系统——在跨用户、跨任务、跨智能体下抽取、复用和分享记忆的能力。

```
个性化记忆 (Personal Memory) + 任务经验 (Task Memory)= agent的记忆管理
```

个性化记忆能够"**理解用户需要**"，任务记忆让agent"**做得更好**"，

---

## 📰 最新动态

- **[2025-09]** 🧪 我们在appworld, bfcl(v3) 以及frozenlake环境验证了记忆抽取与复用在agent中的效果，更多信息请查看 [appworld exp](./cookbook/appworld/quickstart.md), [bfcl exp](./cookbook/bfcl/quickstart.md) & [frozenlake exp](./cookbook/frozenlake/quickstart.md)
- **[2025-09]** 🎉 ReMe(formerly [MemoryScope](./memoryscope/README.md)) v1.0 正式发布，整合任务经验与个人记忆。 如果想使用原始的memoryscope项目，你可以在[MemoryScope](./memoryscope)找到
- **[2025-08]** 🚀 MCP协议支持已上线！→ [快速开始指南](./doc/mcp_quick_start.md)
- **[2025-07]** 📚 完整文档和快速开始指南发布
- **[2025-06]** 🚀 多后端向量存储支持 (Elasticsearch & ChromaDB) -> [快速开始指南](./doc/vector_store_api_guide.md)
- **[2024-09]** 🧠 MemoryScope v0.1.1.0 发布，个性化和时间感知的记忆存储与使用

---

## ✨ 架构设计

### 🎯 双模记忆系统

ReMe整合两种互补的记忆能力：

#### 🧠 **任务经验 (Task Memory/Experience)**
跨智能体复用的程序性知识
- **成功模式识别**：识别有效策略并理解其根本原理
- **失败分析学习**：从错误中学习，避免重复同样的问题
- **对比模式**：不同采样轨迹通过对比得到更有价值的经验
- **验证模式**：经过验证模块确认抽取记忆的有效性

你可以从[task memory](./doc/task_memory/task_memory.md)了解更多如何使用task memory的方法

#### 👤 **个人记忆 (personal memory)**
特定用户的情境化记忆
- **个体偏好**：用户的习惯、偏好和交互风格
- **情境适应**：基于时间和上下文的智能记忆管理
- **渐进学习**：通过长期交互逐步建立深度理解
- **时间感知**：检索和整合时都具备时间敏感性

- 你可以从[personal](./doc/personal_memory/personal_memory.md)了解更多如何使用personal memory的方法


---

## 🛠️ 安装

### 从PyPI安装（推荐）
```bash
pip install reme-ai
```

### 从源码安装
```bash
git clone https://github.com/modelscope/ReMe.git
cd ReMe
pip install .
```

### 环境配置

复制  `example.env` 为 .env并修改其中对应参数：

```bash
# 必需：LLM API配置
FLOW_LLM_API_KEY=sk-xxxx
FLOW_LLM_BASE_URL=https://xxxx/v1

# 必需：嵌入模型配置  
FLOW_EMBEDDING_API_KEY=sk-xxxx
FLOW_EMBEDDING_BASE_URL=https://xxxx/v1
```

---

## 🚀 快速开始

### HTTP服务启动
```bash
reme \
  backend=http \ 
  http.port=8001 \
  llm.default.model_name=qwen3-30b-a3b-thinking-2507 \
  embedding_model.default.model_name=text-embedding-v4 \
  vector_store.default.backend=local
```

### MCP服务器支持
```bash
reme \
  backend=mcp \
  mcp.transport=stdio \
  llm.default.model_name=qwen3-30b-a3b-thinking-2507 \
  embedding_model.default.model_name=text-embedding-v4 \
  vector_store.default.backend=local
```

### 核心API使用

#### 任务经验管理
```python
import requests

# 经验总结器：从执行轨迹学习
response = requests.post("http://localhost:8002/summary_task_memory", json={
    "workspace_id": "task_workspace",
    "trajectories": [
        {"messages": [{"role": "user", "content": "帮我制定项目计划"}], "score": 1.0}
    ]
})

# 经验检索器：获取相关经验
response = requests.post("http://localhost:8002/retrieve_task_memory", json={
    "workspace_id": "task_workspace",
    "query": "如何高效管理项目进度？",
    "top_k": 1
})
```

#### 个人记忆管理  
```python
# 记忆整合：从用户交互中学习
response = requests.post("http://localhost:8002/summary_personal_memory", json={
    "workspace_id": "task_workspace",
    "trajectories": [
        {"messages":
            [
                {"role": "user", "content": "我喜欢早上喝咖啡工作"},
                {"role": "assistant", "content": "了解，您习惯早上用咖啡提神来开始工作"}
            ]
        }
    ]
})

# 记忆检索：获取个人记忆片段
response = requests.post("http://localhost:8002/retrieve_personal_memory", json={
    "workspace_id": "task_workspace",
    "query": "用户的工作习惯是什么？",
    "top_k": 5
})
```

---

## 📦 即用型经验库

ReMe提供预构建的经验库，智能体可以立即使用经过验证的最佳实践：

### 可用经验库
- **`appworld_v1.jsonl`**：Appworld智能体交互的记忆库，涵盖复杂任务规划和执行模式
- **`bfcl_v1.jsonl`**：BFCL工具调用的工作记忆库

### 快速使用
```python
# 加载预构建经验
response = requests.post("http://localhost:8002/vector_store", json={
    "workspace_id": "appworld_v1", 
    "action": "load",
    "path": "./library/"
})

# 查询相关经验
response = requests.post("http://localhost:8002/retrieve_task_memory", json={
    "workspace_id": "appworld_v1",
    "query": "如何导航到设置并更新用户资料？",
    "top_k": 1
})
```

## 🧪 实验

### 🌍 Appworld 实验

我们在 Appworld 上使用 qwen3-8b 测试 ReMe：

| 方法                  | pass@1    | pass@2      | pass@4    |
|---------------------|-----------|-------------|-----------|
| 不使用 ReMe (baseline) | 0.083     | 0.140       | 0.228     |
| **使用 ReMe**         |           |             |           |
| w/ memory(直接使用)     | **0.109** | **0.175**   | **0.281** |

Pass@K 衡量的是在生成的 K 个样本中，至少有一个成功完成任务（score=1）的概率。  
当前实验使用的是一个内部的 AppWorld 环境，可能存在轻微差异。

你可以在 [quickstart.md](cookbook/appworld/quickstart.md) 中找到复现实验的更多细节。


### 🧊 Frozenlake 实验

|                                          不使用memory                                          |                                          使用memory                                           |
|:-------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| <p align="center"><img src="doc/figure/frozenlake_failure.gif" alt="GIF 1" width="30%"></p> | <p align="center"><img src="doc/figure/frozenlake_success.gif" alt="GIF 2" width="30%"></p> |

我们在 100 个随机 frozenlake 地图上使用 qwen3-8b 进行测试：

| 方法                  | pass rate      | 
|---------------------|----------------|
| 不使用 ReMe (baseline) | 0.66           | 
| **使用 ReMe**         |                |
| w/ memory (直接使用)    | 0.72 **(+9.1%)** |

你可以在 [quickstart.md](cookbook/frozenlake/quickstart.md) 中找到复现实验的更多细节。

### 🔧 BFCL-V3 实验

我们在 BFCL-V3 multi-turn-base (随机划分50train/150val) 上使用 qwen3-8b 测试 ReMe：

| 方法                  | pass@1    | pass@2      | pass@4    |
|---------------------|-----------|-------------|-----------|
| 不使用 ReMe (baseline) | 0.2472     | 0.2733       | 0.2922     |
| **使用 ReMe**         | 0.3061 **(+5.89%)** | 0.3500 **(+7.67%)**   | 0.3888 **(+9.66%)** |

## 📚 相关资源

- **[快速开始](./cookbook/simple_demo)**：通过实际示例快速上手
- **[向量存储设置](./doc/vector_store_api_guide.md)**：配置本地/向量数据库以及使用 
- **[mcp指南](./doc/mcp_quick_start.md)**：创建mcp服务
- **链路说明**: 个性化记忆与任务记忆中分别使用的算子及其含义可以分别在 [personal memory](./doc/personal_memory) 与 [task memory](./doc/task_memory)中找到，你可以修改config以自定义链路
- **[示例集合](./cookbook)**：实际用例和最佳实践

---

## 🤝 贡献

我们相信最好的记忆系统来自集体智慧。欢迎贡献：

### 代码贡献
- 新操作和工具开发
- 后端实现和优化
- API增强和新端点

### 文档改进
- 使用示例和教程
- 最佳实践指南

---

## 📄 引用

```bibtex
@software{ReMe2025,
  title = {ReMe: Memory Framework for AI Agent},
  author = {The ReMe Team},
  url = {https://github.com/modelscope/ReMe},
  year = {2025}
}
```

---

## ⚖️ 许可证

本项目采用Apache License 2.0许可证 - 详情请参阅[LICENSE](./LICENSE)文件。

---