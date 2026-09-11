# evals —— mineLangChain agent 评测

按「测试目标」拆成四个独立 CLI，覆盖「效果质量」和「运行性能」两个维度：

| 子命令 | 测什么 | 花不花钱 | 前置 |
|---|---|---|---|
| `evals.retrieval` | 混合检索召回质量（recall@k / MRR） | **0**，本地 bge+BM25 | 跑过 `ingest_obsidian_notes.py` |
| `evals.agent` | 端到端效果回归（工具选择 / grounding / 结构化 / 记忆） | 真实 token | `.env` 的 API key |
| `evals.memory` | 重复相近问题时会不会调长期记忆省 token（带基线对比） | 真实 token | `.env` 的 API key |
| `evals.latency` | 单轮延迟 / token（成本代理）/ 冷启动 | 真实 token | `.env` 的 API key |

产物统一写 `evals/results/<模块>/<时间戳>.jsonl`（已 gitignore）。golden 标签是**可编辑资产**，请随语料/需求增删。

## 离线评测（推荐先跑，零成本）

```bash
# 检索质量：13 条 golden 查询 → recall@1/3/5 + MRR
uv run python -m evals.retrieval

# 只跑前 5 条 / 把融合 top 调大算更深的 recall 曲线
uv run python -m evals.retrieval --limit 5 --top 20
```

看两点：
- 汇总行 `平均 recall@3 / MRR`——检索差，端到端答案大概率也差，先修这里；
- 每条 `完全没召回到相关文件` 的 query——通常是标签标错（语料确实没这内容）或真漏检。

标签文件 `evals/golden_retrieval.json`：`relevant` 填期望命中的**文件名**（按 metadata source 的 basename 精确匹配）。改完重跑即可。

## 真实模型评测（要 token / 网络 / minimax key）

```bash
# 先预览，不调模型、不扣费
uv run python -m evals.agent --dry
uv run python -m evals.memory --dry
uv run python -m evals.latency --dry --cold

# 实跑
uv run python -m evals.agent
uv run python -m evals.memory
uv run python -m evals.latency --cold --iter 3
```

场景/判据在 `evals/agent_scenarios.json`（建议随能力演进增删）。判据默认是**确定性**的：
调用没调该调的工具、结构化 schema 是否成立、`sources` 是否被 `verify_rag_sources` 判为伪造。
自由文本关键词判据只是弱信号，别当成硬通过。

### 接线 LangSmith（可选）

设好任一即可自动开启 tracing（每条 run 留 trace，能看逐节点延迟/工具调用/token）：

```bash
# Windows PowerShell
$env:LANGSMITH_API_KEY="lsv2_..."
# 或 .env 里写 LANGSMITH_API_KEY=...（各 runner 开头的 bootstrap_runtime() 会读）
```

运行时首行会打印 `LangSmith tracing: ON/OFF`。

## 引导（bootstrap）

`import agent` 不再偷偷 `load_dotenv()`，所以四个 runner 都在 `main()` 开头调
`evals.lib.bootstrap_runtime()`（内部就是 `agent.bootstrap()`：.env → HF 缓存 → 日志）。
新增 runner 时别忘这一行，否则 `.env` 里的 key 读不到。

身份也统一走 `evals.lib.identity_context(run_id, user, thread, role="user")`：
它把 `run_id` 编进 `user_id`（`eval-<user>-<run_id>`）保证多次 run 不互相污染记忆，
并显式带上 `role` —— ACL 是 fail-closed 的，不设上下文就一条文档也拿不到。

## 角色与 ACL

`evals.retrieval` 跑的是**故意未挂 ACL** 的底层 retriever（评估检索能力本身，
不受身份策略影响；要是包了 ACLRetriever，没上下文的评测会直接拿到空集）。
`evals.agent` / `evals.memory` 走完整 agent，retriever
被 `ACLRetriever` 包了一层，按当前请求的 `role` 过滤：

- `evals.agent` 默认每个场景用 `role="user"`（除非 scenario JSON 里显式 `"role": "admin"`）
- `evals.memory` 的 with_memory 变体用 `admin`、baseline 用 `user`，对照"有记忆工具 + 全集 RAG" vs "无记忆 + 受限 RAG"的 token 差异
- `evals.latency` 默认 `admin`（不挡 RAG 看全库，看纯延迟）

记忆 ACL 不会影响评测——长期记忆本来就按 `user_id`（eval 唯一）隔离，跟 role 正交。

## 记忆后端（会影响评测）

agent 默认把短期/长期记忆写 SQLite（`data/memory/`），所以**评测会真的落盘**。
为了不让多次 run 互相污染，各 runner 都用「run 唯一」的用户/thread 前缀（`eval-<user>-<run_id>`）。
想完全退出磁盘（更快、更干净）：

```bash
# PowerShell
$env:AGENT_MEMORY_BACKEND="memory"
# bash
export AGENT_MEMORY_BACKEND=memory
```

## 已知限制（别误读结果）

- 检索评测**按文件名判命中**：同文件多 chunk 命中只算一次；跨文件相关的 query 需要 label 自己写全 relevant。
- 端到端场景默认每条构建一个独立 agent（隔离记忆），较慢；判定依赖 minimax 模型的实际发挥，单次跑可能抖动，**要结论请多跑几次看趋势**。
- 记忆探针是「观察 + 基线对比」，不是硬 pass/fail——重点看 `with_memory` 是否调了 `get_user_preference`、均问答输出 token 是否比 baseline 低。
- 延迟是单机/单进程参考值（InMemory 态），不能代表生产并发吞吐；要生产数字请先把记忆持久化 + 服务化做完再压测。

## 想加评测？

每类都是一个独立模块，照现有模式加即可：
- 检索：往 `golden_retrieval.json` 加 case；
- 效果：往 `agent_scenarios.json` 加 scenario（新能力就新建一个条目）；
- 更硬核（Harbor 式冻结环境 + 独立 verifier）：`agent` / `memory` 的 runner 结构已按「场景 + 判据 + 报告」分好，可逐步迁移。
