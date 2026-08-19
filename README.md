# mineLangChain

LangChain 学习示例项目。基于 LangChain 1.x，使用 minimax 的 Anthropic 兼容代理调用 Claude 系列模型。

## 功能特性

本项目实现了 LangChain 官方文档中介绍的三类**上下文管理**：

| 上下文类型 | 实现方式 |
|---|---|
| **短期记忆** | `InMemorySaver` checkpointer + `thread_id` 实现多轮对话 |
| **消息摘要** | `SummarizationMiddleware` 超 token 阈值时压缩老消息（默认 2000） |
| **长期记忆** | `InMemoryStore` + `save_user_preference` / `get_user_preference` 工具 |

## 内置中间件

`agent/builder.py` 在 `create_agent` 时注册了三个中间件：

| 中间件 | 作用 | 配置 |
|---|---|---|
| `ModelCallLimitMiddleware` | 防止 agent 死循环 | `thread_limit=15`, `run_limit=20`, `exit_behavior="end"` |
| `HumanInTheLoopMiddleware` | 敏感工具调用前询问用户 | `slow_lookup` 需要 `approve` / `edit` / `reject` 决策 |
| `make_summarization_middleware()` | token 超阈值时压缩老消息 | `trigger=("tokens", 2000)`, `keep=("tokens", 400)` |

## 项目结构

```
mineLangChain/
├── agent/                       ← Agent 包（构造 + 配置）
│   ├── __init__.py              # 暴露 build_agent
│   ├── builder.py               # 组装层：合成 LLM + context + middleware + tools
│   ├── llm.py                   # ChatAnthropic 工厂（minimax 代理）
│   ├── tools.py                 # 演示型工具（slow_lookup，用于演示流式）
│   ├── context/                 ← 上下文管理（非中间件）
│   │   ├── __init__.py
│   │   ├── short_term.py        # 短期记忆：checkpointer
│   │   └── long_term.py         # 长期记忆：Store + preference 工具
│   └── middleware/              ← 中间件（每个中间件一个文件）
│       ├── __init__.py
│       ├── model_call_limit.py  # 模型调用上限，防死循环
│       ├── human_in_the_loop.py # 人在回路，slow_lookup 前暂停审批
│       └── summarization.py     # 消息摘要，超 token 阈值时压缩老消息
├── main.py                      # 入口：流式多轮对话（订阅三种 stream_mode）
├── demos/                       # 一键演示脚本
│   └── long_conversation.py     # 假长对话，自动触发 SummarizationMiddleware
├── .env                         # API key 等本地配置（已加入 .gitignore）
├── pyproject.toml
└── uv.lock
```

`main.py` 只 `from agent import build_agent`，agent 包的内部细节对入口透明。

## 三种数据流（stream_mode）

`main.py` 同时订阅五种 stream_mode：

| 模式 | 触发 | 终端表现 |
|---|---|---|
| `messages` | LLM 逐 token 输出 | `<<< 你好！很高...` 打字机效果 |
| `custom` | 工具内 `get_stream_writer()` 推送 | `⚙ [custom] {'event': 'progress', ...}` |
| `updates` | 每个 graph 节点执行完（仅 `--debug` 开启时打印） | `↳ [model] AIMessage: 你好！...` |
| `values` | 每个节点执行完的 state 完整快照（仅 `--debug`） | `◇ [values] 当前消息数：N` |
| `events` | 底层图事件（仅 `--debug`） | `◆ [events] on_tool_start` |

## HITL（Human-in-the-Loop）

当 agent 调用 `slow_lookup` 时，HumanInTheLoopMiddleware 会暂停执行，
向终端打印工具名/参数/描述，询问你的决策：

```
────────────────────────────────────────────────────────────
⚠️  工具调用需要你的确认（Human-in-the-Loop）：
────────────────────────────────────────────────────────────

[1] 工具名: slow_lookup
    描述:   slow_lookup 是一个演示用的慢速查询，会 sleep 5 次...
    参数:   {
        "query": "北京"
}
    决策 → (a)pprove / (e)dit / (r)eject: 
```

输入 `a` 批准、`e` 编辑参数、`r` 拒绝。决策通过 `Command(resume={...})` 恢复 agent 执行。

试试：

```bash
# 普通模式
uv run python main.py

# 调试模式（额外打印 updates / values / events）
uv run python main.py --debug
```

进入对话后输入：

- `你好` —— 看 token 流（messages）
- `请用 slow_lookup 查一下北京` —— 看工具推送的进度（custom）+ HITL 弹窗
- `--debug` 启动后任何输入 —— 看每步 state 增量（updates / values / events）

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