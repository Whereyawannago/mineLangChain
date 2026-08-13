# mineLangChain

LangChain 学习示例项目。基于 LangChain 1.x，使用 minimax 的 Anthropic 兼容代理调用 Claude 系列模型。

## 功能特性

本项目实现了 LangChain 官方文档中介绍的三类**上下文管理**：

| 上下文类型 | 实现方式 |
|---|---|
| **短期记忆** | `InMemorySaver` checkpointer + `thread_id` 实现多轮对话 |
| **消息裁剪** | `@before_model` 中间件 + `trim_messages` 按 token 阈值裁剪 |
| **长期记忆** | `InMemoryStore` + `save_user_preference` / `get_user_preference` 工具 |

## 项目结构

```
mineLangChain/
├── agent.py     # Agent 构造：LLM、工具、消息裁剪中间件、checkpointer / store
├── main.py      # 入口：仅包含多轮对话循环
├── .env         # API key 等本地配置（已加入 .gitignore）
├── pyproject.toml
└── uv.lock
```

`build_agent()` 在 `agent.py` 里定义，`main.py` 只 `from agent import build_agent` 然后跑循环。

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/Whereyawannago/mineLangChain.git
cd mineLangChain
uv sync
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint
ANTHROPIC_MODEL=claude-sonnet-4-5
```

> 本项目默认使用 minimax 的 Anthropic 兼容代理（`https://api.minimaxi.com/anthropic`），您也可以换成任意 Anthropic 兼容端点。

### 3. 运行

```bash
uv run python main.py
```

进入交互式多轮对话：

```
>>> 你好，我叫 Alice
<<< 你好 Alice！很高兴认识你~

>>> 我叫什么？
<<< 你叫 Alice。
```

输入 `quit` / `exit` / `退出` 结束对话。

## 关键依赖

- `langchain >= 1.3.15` — 1.x 版本，使用 `create_agent` API
- `langchain-anthropic >= 1.5.5` — Anthropic Chat 模型集成
- `deepagents >= 0.7.5`
- `python-dotenv` — `.env` 文件加载

## 学习资源

本项目用到了 LangChain Skills（位于 `.claude/skills/` 下，但被 `.gitignore` 排除）。如需恢复：

```bash
npx -y @skills add langchain-ai/langchain-skills --agent claude-code --skill '*' --yes
```

或访问官方文档：[docs.langchain.com](https://docs.langchain.com)

## License

MIT