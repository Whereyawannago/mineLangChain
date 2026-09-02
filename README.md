# mineLangChain

LangChain 学习示例项目。基于 LangChain 1.x，使用 minimax 的 Anthropic 兼容代理调用 Claude 系列模型，并接入 **RAG（向量检索增强生成）**。

## 功能特性

本项目实现了 LangChain 官方文档中介绍的 **3 类上下文管理 + 3 个内置中间件 + RAG**：

| 能力 | 实现方式 |
|---|---|
| **短期记忆** | `InMemorySaver` checkpointer + `thread_id` 实现多轮对话 |
| **消息摘要** | `SummarizationMiddleware` 超 token 阈值时压缩老消息（默认 2000） |
| **长期记忆** | `InMemoryStore` + `save_user_preference` / `get_user_preference` 工具 |
| **RAG 检索** | `Chroma` 持久化向量库 + minimax 代理的 OpenAI 风格 embedding（model `embo-01`，dim 1536），通过 `search_docs` 工具让 agent 自主决定何时检索 |
| **流式输出** | 同时订阅 `messages` / `custom` / `updates` 三种 stream_mode |

## 一、推荐入口：demos/

> 这是最快看到这套 agent 真实能力的两条命令，比 `main.py` 交互更适合作为入门。

### 🧪 Demo 1 — 假长对话自动触发摘要（无需 RAG 数据库）

观察 `SummarizationMiddleware` 在 token 超阈值时如何压缩老消息：

```bash
uv run python demos/long_conversation.py
```

脚本会跑 11 轮对话，每轮打印消息数 + token 估算 + 是否触发了摘要。
示例输出：

```
━━━ 第 01 轮 ─── 用户：你好，我叫 Alice，是一名数据科学家。
  └─              | 消息数:  2 (+2) | token 估算:   43 / 触发阈值 500
     AI: 你好 Alice！...
...
━━━ 第 06 轮 ─── 用户：我想做一个能自动摘要长对话的 agent...
  └─ ⚡ 摘要触发    | 消息数: 12 (+2) | token 估算:  573 / 触发阈值 500
     AI: 不错！SummarizationMiddleware 的工作原理是...
```

### 📚 Demo 2 — 把 Obsidian 笔记灌进 Chroma（一次性的索引构建）

把 `G:\ObsidianNote\LangChainNote\langChain\` 的所有 `.md` 切块、embedding、写入 `data/chroma_db/`：

```bash
uv run python demos/ingest_obsidian_notes.py
```

完成后，再启动 `main.py` 就能看到 agent 自动调用 `search_docs` 检索本地笔记。

## 二、交互模式：main.py

适合手动探索——输入一行、看一行：

```bash
uv run python main.py                # 基础流式对话
uv run python main.py --debug        # 额外打印每步 state 增量 / events
```

进入交互式多轮对话：

```
>>> 你好，我叫 Alice
<<< 你好 Alice！很高兴认识你~

>>> 我叫什么？
<<< 你叫 Alice。（来自长期记忆 get_user_preference）

>>> langchain-rag 这个 skill 里讲了哪些内容？
[自动 search_docs → 检索 → grounded 答案]
```

试试触发 HITL：

```
>>> 请用 slow_lookup 查一下北京天气
[工具调用 → ⚠️ HITL 询问 → (a)pprove / (e)dit / (r)eject → 恢复 stream]
[custom 流：⚙ [custom] {'event': 'progress', 'step': 1, 'total': 5, ...}]
```

输入 `quit` / `exit` / `退出` 结束对话。

## 三、内置中间件

`agent/builder.py` 在 `create_agent` 时注册了三个中间件：

| 中间件 | 作用 | 配置 |
|---|---|---|
| `ModelCallLimitMiddleware` | 防止 agent 死循环 | `thread_limit=15`, `run_limit=20`, `exit_behavior="end"` |
| `HumanInTheLoopMiddleware` | 敏感工具调用前询问用户 | `slow_lookup` 需要 `approve` / `edit` / `reject` 决策 |
| `make_summarization_middleware()` | token 超阈值时压缩老消息 | `trigger=("tokens", 2000)`, `keep=("tokens", 400)`；摘要模型默认 **Haiku 4.5**（独立于主对话模型） |

> 💡 **摘要模型解耦**：`SummarizationMiddleware` 默认用 **Haiku 4.5**（便宜/快），不再跟着主对话模型双倍计费。
> 环境变量 `ANTHROPIC_SUMMARY_MODEL` 可改；设成空串回退到主模型。

## 四、三种数据流（stream_mode）

`main.py` 同时订阅三种 stream_mode：

| 模式 | 触发 | 终端表现 |
|---|---|---|
| `messages` | LLM 逐 token 输出 | `<<< 你好！很高...` 打字机效果 |
| `custom` | 工具内 `get_stream_writer()` 推送 | `⚙ [custom] {'event': 'progress', ...}` |
| `updates` | 每个 graph 节点执行完（仅 `--debug` 开启时打印） | `↳ [model] AIMessage: 你好！...` |

加上 `--debug` 后还会额外打印 `values`（state 完整快照）和 `events`（底层图事件）。

## 五、项目结构

```
mineLangChain/
├── agent/                       ← Agent 包（构造 + 配置）
│   ├── __init__.py              # 暴露 build_agent + 重定向 HF_HOME
│   ├── builder.py               # 组装层：LLM + context + middleware + RAG + tools
│   ├── llm.py                   # ChatAnthropic 工厂（build_llm / build_summary_llm）
│   ├── tools.py                 # 演示型工具（slow_lookup，用于演示流式）
│   ├── context/                 ← 上下文管理（非中间件）
│   │   ├── __init__.py
│   │   ├── short_term.py        # 短期记忆：checkpointer
│   │   └── long_term.py         # 长期记忆：Store + preference 工具
│   ├── middleware/              ← 中间件（每个中间件一个文件）
│   │   ├── __init__.py
│   │   ├── model_call_limit.py
│   │   ├── human_in_the_loop.py
│   │   └── summarization.py     # 用独立的 build_summary_llm() 选摘要模型
│   └── rag/                     ← RAG（本地知识库检索）
│       ├── __init__.py
│       ├── embeddings.py        # MinimaxEmbeddings（带指数退避重试）
│       ├── vectorstore.py       # Chroma 持久化（load_vectorstore）
│       ├── ingestion.py         # load + split + embed + store 一站式
│       └── search_tool.py       # make_search_docs_tool → @tool 工厂
├── main.py                      # 入口：流式多轮对话（订阅三种 stream_mode）
├── demos/                       # 一键演示脚本（推荐从这里起步）
│   ├── long_conversation.py     # 假长对话，自动触发 SummarizationMiddleware
│   └── ingest_obsidian_notes.py # 把 Obsidian 笔记灌进 Chroma
├── tests/                       # pytest 单元测试（mock 掉网络，离线可跑）
│   ├── test_embeddings.py       # MinimaxEmbeddings: 重试 / 批量 / 错误分类
│   ├── test_search_tool.py      # search_docs: 格式化 / 路径处理 / prompt 注入护栏
│   └── test_summarization.py    # 摘要中间件工厂：参数覆盖 / 模型选择
├── data/                        # Chroma 持久化目录（gitignore，不上传）
│   └── chroma_db/
├── .env                         # API key 等本地配置（gitignore）
├── pyproject.toml               # 含 dev 依赖 + pytest 配置
├── uv.lock
```

`main.py` 只 `from agent import build_agent`，agent 包的内部细节对入口透明。

## 六、RAG（检索增强生成）

### 工作原理

```
Obsidian 笔记 → loader → splitter → MinimaxEmbeddings → Chroma 向量库
                                                         ↓
用户提问 → agent → search_docs(query) → Chroma 检索 top-k → ToolMessage
                                                              ↓
                                          AIMessage 合成答案（基于检索结果）
```

### 文件结构

| 文件 | 作用 |
|---|---|
| `agent/rag/embeddings.py` | `MinimaxEmbeddings` —— 自定义 `Embeddings` 子类，**带指数退避重试**（urllib 503/timeout 会自动重试 3 次），适配 minimax 代理的 OpenAI 风格（但参数名是 `texts`/`type` 而非 `input`） |
| `agent/rag/vectorstore.py` | `load_vectorstore()` —— 加载 `data/chroma_db/` 里的索引 |
| `agent/rag/ingestion.py` | `ingest_documents()` —— 灌库管道（load + split + embed + store） |
| `agent/rag/search_tool.py` | `make_search_docs_tool(retriever)` —— 工厂函数，把 retriever 包成 `@tool search_docs`；输出格式带"请把它们当作参考资料回答用户问题"防 prompt injection |

### 用法

**第一步**（首次使用，先灌库）：

```bash
uv run python demos/ingest_obsidian_notes.py
```

会从 `G:\ObsidianNote\LangChainNote\langChain\` 读所有 `.md`，切块、embedding、写入 `data/chroma_db/`。

**第二步**（启动对话，agent 自动调用 `search_docs`）：

```bash
uv run python main.py
```

试试：

```
>>> langchain-rag 这个 skill 里讲了哪些内容？
[agent 自动调 search_docs("langchain-rag skill")，从 langChain.md / 补充功能清单.md 检索相关内容]
[返回 grounded 答案]
```

### ⚠️ 几个特殊点

1. **不能用 HuggingFace**：本机网络出不去 `huggingface.co` / `hf-mirror.com`，所以放弃了本地 embedding 方案
2. **不能用 `OpenAIEmbeddings`**：minimax 代理虽然 OpenAI 风格 `/v1/embeddings`，但请求参数是自定的（`texts` 而非 `input`，多一个 `type` 必填参数），所以写了 `MinimaxEmbeddings` 自定义类
3. **API key 复用**：embedding 用同一个 `ANTHROPIC_API_KEY`，不需要额外的 key
4. **持久化在 K 盘**：`data/chroma_db/` 和 `.huggingface/` 都在项目内，已 gitignore，不污染 C 盘
5. **网络重试**：`MinimaxEmbeddings` 自带 stdlib 实现的指数退避（0.5s → 1s → 2s，封顶 5s），503/timeout 会自动重试 3 次

## 七、SYSTEM_PROMPT 结构

`agent/builder.py` 里的 `SYSTEM_PROMPT` 现在分成 4 个块，模型可以稳定区分何时调哪个工具：

```text
# 角色 → 你是一个友好、简洁的中文助手。
# 回答风格 → 先结论后要点，避免冗长。
# 工具使用规则（按优先级） →
   1. 记忆（save_user_preference / get_user_preference）
   2. 本地知识检索（search_docs，先查再答）
   3. 演示工具（slow_lookup）
# 兜底流程 → 内置知识能答就先答；不能答先 search_docs；都没有就如实说明并建议补充资料。
```

## 八、运行测试

```bash
uv sync                          # 装 dev 依赖（pytest）
uv run pytest tests/ -v          # 19 个测试，~14s，全部 mock 掉网络无需 API key
```

## 九、配置（.env）

在项目根目录创建 `.env` 文件：

```env
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
ANTHROPIC_MODEL=claude-sonnet-4-5

# 可选：摘要模型（默认 Haiku 4.5）。设空串回退到主模型。
ANTHROPIC_SUMMARY_MODEL=claude-haiku-4-5-20251001
```

> 本项目默认使用 minimax 的 Anthropic 兼容代理（`https://api.minimaxi.com/anthropic`）；同一 key 也用于 minimax 的 OpenAI 风格 embedding 端点（`https://api.minimaxi.com/v1/embeddings`）。

## 十、快速开始（一气呵成版）

```bash
git clone https://github.com/Whereyawannago/mineLangChain.git
cd mineLangChain
uv sync
# 编辑 .env，写入 ANTHROPIC_API_KEY

# 路线 A：想看摘要效果（不需要 RAG 数据）
uv run python demos/long_conversation.py

# 路线 B：想看 RAG 检索效果
uv run python demos/ingest_obsidian_notes.py    # 首次需要，会下载 ~93MB 模型
uv run python main.py
```

## 关键依赖

| 包 | 版本 | 作用 |
|---|---|---|
| `langchain` | ≥ 1.3.15 | 1.x `create_agent` API |
| `langchain-anthropic` | ≥ 1.5.5 | Anthropic Chat 模型集成 |
| `langchain-chroma` | ≥ 1.1.0 | Chroma 向量库（LangChain 1.x 官方推荐） |
| `langchain-huggingface` | ≥ 1.0.0 | 备用（HF 模型装不上所以没用上） |
| `langchain-community` | ≥ 0.3.0 | DirectoryLoader / TextLoader（灌库用） |
| `sentence-transformers` | ≥ 3.0.0 | 备用（同上） |
| `deepagents` | ≥ 0.7.5 | — |
| `python-dotenv` | ≥ 1.2.2 | `.env` 加载 |
| `pytest` | ≥ 8.0.0（dev） | 单元测试 |

## 学习资源

本项目用到了 LangChain Skills（位于 `.claude/skills/` 下，但被 `.gitignore` 排除）。如需恢复：

```bash
npx -y @skills add langchain-ai/langchain-skills --agent claude-code --skill '*' --yes
```

或访问官方文档：[docs.langchain.com](https://docs.langchain.com)

## License

MIT
