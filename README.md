# mineLangChain

LangChain 学习示例项目。基于 LangChain 1.x，通过 Anthropic 兼容端点调用 Claude 系列模型，并接入 **RAG（混合检索增强生成）** 与 **结构化输出**。

## 功能特性

本项目实现了 LangChain 官方文档中介绍的 **3 类上下文管理 + 3 个内置中间件 + RAG + 结构化输出**：

| 能力 | 实现方式 |
|---|---|
| **短期记忆** | **SqliteSaver** checkpointer（`data/memory/checkpoints.sqlite`，跨进程存活）+ `thread_id` 多轮对话 |
| **消息摘要** | `SummarizationMiddleware` 超 token 阈值时压缩老消息（默认 2000） |
| **长期记忆** | **SqliteStore**（`data/memory/store.sqlite`）+ `save_user_preference` / `get_user_preference` 工具，按 `user_id` 隔离 |
| **RAG 检索** | Chroma 向量库 + **混合检索**（dense + BM25 + RRF 融合），本地 bge 中文 embedding，通过 `search_docs` 工具让 agent 自主决定何时检索 |
| **结构化输出** | `ToolStrategy` + Pydantic schema，`result["structured_response"]` 直接是 schema 实例；`verify_rag_sources()` 后校验引用来源不被模型伪造 |
| **流式输出** | 同时订阅 `messages` / `custom` / `updates` / `values` / `events` 五种 stream_mode |

> 记忆默认持久化到 SQLite（`data/`，已 gitignore），**进程重启不丢**。想退回内存态：
> `AGENT_MEMORY_BACKEND=memory`。多实例生产部署建议换 PostgresSaver / PostgresStore（见文末「生产化」）。

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

把 `G:\ObsidianNote\` 下所有 `.md` 切块、embedding、写入 `data/chroma_db/`：

```bash
uv run python demos/ingest_obsidian_notes.py
```

完成后，再启动 `main.py` 就能看到 agent 自动调用 `search_docs` 检索本地笔记。

### 🧩 Demo 3 — 结构化输出

让 agent 的回复强制收敛到 Pydantic schema，并演示 `verify_rag_sources()` 剔除伪造引用：

```bash
uv run python demos/structured_output.py
```

三段分别展示 `ChatReply`（最小形态）、`WeatherReport`（业务形态）、`RAGAnswer`（检索增强形态，需先跑过 Demo 2）。返回值不再是自由文本字符串，而是可以直接 `.字段` 取值的 schema 实例。

## 二、交互模式：main.py

适合手动探索——输入一行、看一行：

```bash
uv run python main.py                          # 基础流式对话（默认用户 local-dev）
uv run python main.py --debug                  # 额外打印每步 state 增量 / events
uv run python main.py --user alice --session s1  # 指定身份；同 user+session 可跨进程恢复对话
```

> 身份与会话：`--user` 决定长期记忆的命名空间（每个用户一份偏好），
> `--session` 决定短期记忆的 thread_id。**同一 `--user --session` 再次启动会接着上次聊**
> （记忆持久化在 `data/memory/`）；不指定 session 则随机，等于开新会话。

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

## 四、结构化输出

普通对话里 `result['messages'][-1].content` 是自由文本；`build_structured_agent()` 构造的 agent 会用 `response_format=ToolStrategy(schema)` 把输出**强制收敛**到 Pydantic schema，业务代码直接 `.字段` 取值，不再写正则 / JSON 解析。

```python
from agent import build_structured_agent
from agent.structured import WeatherReport

agent = build_structured_agent(WeatherReport, include_rag=False)
result = agent.invoke({"messages": [{"role": "user", "content": "上海天气？"}]})
report = result["structured_response"]   # WeatherReport 实例
print(report.city, report.temperature)
```

**为什么是 ToolStrategy 而不是 JSON mode？** JSON mode 让模型在 content 里吐 JSON 再解析，容易夹带文字、缺字段、类型错乱；ToolStrategy 复用 tool calling 机制，模型"更听话"，返回键直接落在 `result["structured_response"]`。

内置 schema：`ChatReply`（content / confidence）、`WeatherReport`、`RAGAnswer`（answer / sources / confidence）。

### 引用来源后校验（`verify_rag_sources`）

`ToolStrategy` 只保证输出**形状**符合 schema，不保证**内容**可信——模型可能在 `RAGAnswer.sources` 里编造不存在的来源。`verify_rag_sources(result)` 以本轮 `search_docs` 真实返回过的来源为基线，剔除所有不在基线里的路径：

```python
from agent.structured import verify_rag_sources

rag, fabricated = verify_rag_sources(result)
# rag.sources          → 已剔除伪造引用
# fabricated           → 被剔除的来源列表（空 = 全部真实）
```

返回的 `rag` 是清理后的副本，不污染原 `result`。

## 五、数据流（stream_mode）

`main.py` 同时订阅五种 stream_mode：

| 模式 | 触发 | 终端表现 |
|---|---|---|
| `messages` | LLM 逐 token 输出 | `<<< 你好！很高...` 打字机效果 |
| `custom` | 工具内 `get_stream_writer()` 推送 | `⚙ [custom] {'event': 'progress', ...}` |
| `updates` | 每个 graph 节点执行完（仅 `--debug` 开启时打印） | `↳ [model] AIMessage: 你好！...` |
| `values` | 每节点 state 完整快照（`--debug` 时打印） | `◇ [values] 当前消息数：N` |
| `events` | 底层图事件（`--debug` 时打印） | `◆ [events] <node>` |

## 六、项目结构

```
mineLangChain/
├── agent/                       ← Agent 包（构造 + 配置）
│   ├── __init__.py              # 暴露 build_agent / build_structured_agent + 重定向 HF_HOME
│   ├── builder.py               # 组装层：LLM + context + middleware + RAG + tools
│   ├── llm.py                   # ChatAnthropic 工厂（build_llm / build_summary_llm）
│   ├── structured.py            # 结构化输出：schema + ToolStrategy + verify_rag_sources
│   ├── tools.py                 # 演示型工具（slow_lookup，用于演示流式）
│   ├── context/                 ← 上下文管理（持久化 + 身份隔离）
│   │   ├── short_term.py        # 短期记忆：checkpointer（默认 SqliteSaver）
│   │   ├── long_term.py         # 长期记忆：Store（默认 SqliteStore）+ preference 工具
│   │   ├── identity.py          # UserContext / thread_id 生成与解析
│   │   └── backends.py          # 后端选择（sqlite/memory）+ DB 路径
│   ├── middleware/              ← 中间件（每个中间件一个文件）
│   │   ├── model_call_limit.py
│   │   ├── human_in_the_loop.py
│   │   └── summarization.py     # 用独立的 build_summary_llm() 选摘要模型
│   └── rag/                     ← RAG（混合检索）
│       ├── embeddings.py        # HuggingFace bge-small-zh（本地、dim 512、CPU）
│       ├── vectorstore.py       # Chroma 持久化（load_vectorstore）
│       ├── ingestion.py         # load + split + embed + store 一站式
│       ├── search_tool.py       # make_search_docs_tool → @tool 工厂
│       ├── bm25_index.py        # BM25 稀疏索引 + SHA1 漂移自检 + 原子持久化
│       └── hybrid_retriever.py  # dense + sparse → RRF 融合（自研，可单测）
├── main.py                      # 入口：流式多轮对话（订阅五种 stream_mode）
├── demos/                       # 一键演示脚本（推荐从这里起步）
│   ├── long_conversation.py     # 假长对话，自动触发 SummarizationMiddleware
│   ├── ingest_obsidian_notes.py # 把 Obsidian 笔记灌进 Chroma
│   └── structured_output.py     # ToolStrategy 结构化输出 + 引用校验
├── tests/                       # pytest 单元测试（mock 掉模型/网络，离线可跑）
│   ├── conftest.py              # 测试期把记忆后端设为 memory，避免写真实 DB
│   ├── test_bm25_index.py       # BM25 分词 / 持久化 / hash 漂移重建
│   ├── test_embeddings.py       # bge 实例 + embed_query / embed_documents 维度
│   ├── test_hybrid_retriever.py # RRF 融合 / 去重 / 权重 / top_k
│   ├── test_search_tool.py      # search_docs: 格式化 / 路径处理 / 防注入护栏
│   ├── test_identity.py         # UserContext / thread_id / 命名空间隔离 / 后端选择
│   ├── test_persistence.py      # sqlite 持久化：新连接读回 / 用户隔离 / 不撞锁
│   ├── test_structured.py       # schema 校验 / ToolStrategy / verify_rag_sources
│   └── test_summarization.py    # 摘要中间件工厂：参数覆盖 / 模型选择
├── data/                        # 本地持久化目录（gitignore，不上传）
│   ├── chroma_db/               # 向量库
│   ├── bm25_index.pkl / .sha1   # BM25 索引 + 漂移校验
│   └── memory/                  # 记忆：checkpoints.sqlite + store.sqlite
├── .env                         # API key 等本地配置（gitignore）
├── pyproject.toml               # 依赖 + dev 依赖 + pytest 配置
└── uv.lock
```

`main.py` 只 `from agent import build_agent`，agent 包的内部细节对入口透明。

## 七、RAG（混合检索）

### 工作原理

```
Obsidian 笔记 → loader → splitter → 本地 bge embedding → Chroma 向量库（dense）
                                                        + jieba 分词 BM25（sparse）
                                                              ↓
                                                     RRF 融合 → top_k
                                                              ↓
用户提问 → agent → search_docs(query) → HybridRetriever → ToolMessage
                                                              ↓
                                          AIMessage 合成答案（基于检索结果）
```

- **dense 路**：Chroma 持久化向量库，HuggingFace `BAAI/bge-small-zh-v1.5`（dim 512，中文友好，~93MB，本地 CPU）。
- **sparse 路**：BM25 稀疏索引，jieba 分词 + 保留英文标识符（`SummarizationMiddleware` 这类 token 也能命中）。
- **融合**：自研 `HybridRetriever`，Reciprocal Rank Fusion，默认权重 dense 0.6 / sparse 0.4，常数 `c=60`，`top_k=4`；按 `source + 前 200 字符` 去重，能压掉 splitter 产生的近重复 chunk。

### 用法

**第一步**（首次使用，先灌库）：

```bash
uv run python demos/ingest_obsidian_notes.py
```

从 `G:\ObsidianNote\` 递归读所有 `.md`，切块（chunk 500 / overlap 80）、embedding、写入 `data/chroma_db/`，并同步生成 BM25 索引。

**第二步**（启动对话，agent 自动调用 `search_docs`）：

```bash
uv run python main.py
```

试试：

```
>>> langchain 这个 skill 里讲了哪些内容？
[agent 自动调 search_docs，返回带来源的 grounded 答案]
```

### 设计要点 / 已知边界

1. **embedding 是本地模型**：首次灌库会从 HuggingFace（或 hf-mirror）下载 ~93MB 权重到项目内 `.huggingface/`（已 gitignore）。机器无法出网时需提前把模型放进缓存目录。
2. **BM25 索引自动漂移校验**：BM25 索引持久化在 `data/bm25_index.pkl`，启动时用 Chroma 内容算 SHA1 比对，不一致就重建；写入用 `tmp → os.replace` 原子写，防半截文件。
3. **灌库会整库重建**：`ingest_documents` 每次用 `Chroma.from_documents` 会生成新 collection，所以 ingest 脚本先 `shutil.rmtree` 清空旧库保证幂等。
4. **数据在本地**：`data/` 与 `.huggingface/` 都在项目内且已 gitignore，不占 C 盘、不上传仓库。
5. **无相似度阈值**：库里没相关内容时仍会硬回 top-k，靠 prompt 兜底 + `confidence` 字段提示"查不到"——对教学够用，生产可加 score floor。

## 八、SYSTEM_PROMPT 结构

`agent/builder.py` 里的 `SYSTEM_PROMPT` 分块组织，让模型稳定区分何时调哪个工具：

```text
# 角色 → 你是一个友好、简洁的中文助手。
# 回答风格 → 先结论后要点，避免冗长。
# 工具使用规则（按优先级） →
   1. 记忆（save_user_preference / get_user_preference）
   2. 本地知识检索（search_docs，先查再答）
   3. 演示工具（slow_lookup）
# 兜底流程 → 内置知识能答就先答；不能答先 search_docs；都没有就如实说明并建议补充资料。
```

## 九、运行测试

```bash
uv sync                          # 装 dev 依赖（pytest）
uv run pytest tests/ -q          # 65 个测试，全绿；离线可跑（mock 掉模型/网络，记忆走内存后端）
```

> 唯一例外：`tests/test_embeddings.py` 会真的加载一次 bge 模型（首次约 93MB），需要机器能访问 HF Hub 或已缓存。

## 十、配置（.env）

在项目根目录创建 `.env` 文件：

```env
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
ANTHROPIC_MODEL=claude-sonnet-4-5

# 可选：摘要模型（默认 Haiku 4.5）。设空串回退到主模型。
ANTHROPIC_SUMMARY_MODEL=claude-haiku-4-5-20251001
```

> 聊天走 `ANTHROPIC_BASE_URL` 指定的 Anthropic 兼容端点（例：minimax 代理）。embedding 用本地 HuggingFace 模型，**不再需要** embedding 用的远端 API key。若 `ANTHROPIC_BASE_URL` 留空则走官方 Anthropic API。

## 十一、快速开始（一气呵成版）

```bash
git clone <你的仓库地址> mineLangChain
cd mineLangChain
uv sync
# 编辑 .env，写入 ANTHROPIC_API_KEY

# 路线 A：想看摘要效果（不需要 RAG 数据）
uv run python demos/long_conversation.py

# 路线 B：想看 RAG 检索效果
uv run python demos/ingest_obsidian_notes.py    # 首次需要，会下载 ~93MB 模型
uv run python main.py

# 路线 C：想看结构化输出 + 引用校验
uv run python demos/structured_output.py
```

## 关键依赖

| 包 | 版本 | 作用 |
|---|---|---|
| `langchain` | ≥ 1.3.15 | 1.x `create_agent` API |
| `langchain-anthropic` | ≥ 1.5.5 | Anthropic Chat 模型集成 |
| `langchain-chroma` | ≥ 1.1.0 | Chroma 向量库（LangChain 1.x 官方推荐） |
| `langchain-huggingface` | ≥ 1.2.2 | 本地 bge embedding |
| `langchain-community` | ≥ 0.4.2 | DirectoryLoader / TextLoader（灌库） |
| `sentence-transformers` | ≥ 6.0.0 | 本地 embedding 模型的推理后端 |
| `jieba` | ≥ 0.42.1 | BM25 中文分词 |
| `rank-bm25` | ≥ 0.2.2 | BM25Retriever 的稀疏检索实现 |
| `python-dotenv` | ≥ 1.2.2 | `.env` 加载 |
| `pytest` | ≥ 8.0.0（dev） | 单元测试 |

## 学习资源

本项目用到了 LangChain Skills（位于 `.claude/skills/` 下，但被 `.gitignore` 排除）。如需恢复：

```bash
npx -y @skills add langchain-ai/langchain-skills --agent claude-code --skill '*' --yes
```

或访问官方文档：[docs.langchain.com](https://docs.langchain.com)

## 生产化（记忆持久化）

本项目已把「重启即丢」的内存记忆换成持久化后端，但仍是**单机**方案。上线到多实例服务时按下表替换：

| 关注点 | 现在（单机/学习） | 生产（多实例） |
|---|---|---|
| 短期记忆 | `SqliteSaver` → `data/memory/checkpoints.sqlite` | `PostgresSaver`（所有实例共享一个 DB） |
| 长期记忆 | `SqliteStore` → `data/memory/store.sqlite` | `PostgresStore` |
| 用户隔离 | `user_id` 命名空间（`UserContext` 注入） | 同上，`user_id` 取自鉴权层 |
| thread_id | `<user_id>:<session>` 由 `new_thread_id()` 生成 | 由客户端/会话表给出稳定 session_id |

替换只需改 `agent/context/short_term.py` / `long_term.py` 里的工厂函数（各一处），
`identity.py` / `builder.py` / 调用方都不用动：

```python
# agent/context/short_term.py —— 生产分支示意
from langgraph.checkpoint.postgres import PostgresSaver
def create_checkpointer():
    return PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
```

注意：
- `PostgresSaver.setup()` 只在**部署时**跑一次建表，不要放在每次启动；
- `user_id` 必须来自请求上下文，**绝不要**用模块级常量或让会话随机 uuid —— 那正是本项目早期
  `USER_ID = "demo-user"` 的坑（所有用户共用一份偏好）；
- 工具侧读身份统一走 `user_id_from_runtime(runtime)`，不要自己从别处取。

## License

MIT
