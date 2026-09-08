"""Agent 组装层

把 LLM、上下文管理（短期/长期）、中间件（模型调用上限 / HITL / 摘要）以及 RAG 检索工具
组合成最终可调用的 agent。

调用方式：
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "..."}]},
        config={"configurable": {"thread_id": "..."}},
    )
"""

from langchain.agents import create_agent

from .context import (
    create_checkpointer,
    create_store,
    preference_tools,
)
from .llm import build_llm
from .middleware import (
    make_human_in_the_loop_middleware,
    make_model_call_limit_middleware,
    make_summarization_middleware,
)
from .rag import (
    HybridRetriever,
    build_embeddings,
    load_or_build_bm25_retriever,
    load_vectorstore,
    make_search_docs_tool,
)
from .tools import demo_tools

SYSTEM_PROMPT = """\
# 角色
你是一个友好、简洁的中文助手。

# 回答风格
- 按照推理逻辑，列出推理过程。
- 先给一句话结论，再用要点展开。
- 避免冗长；不要重复用户已经说过的话。

# 工具使用规则（按优先级）
1. **记忆**
   - 用户提到他的偏好（名字、语言、称呼等）→ 调用 `save_user_preference` 持久化。
   - 用户问起他之前的偏好 → 调用 `get_user_preference` 查询。
2. **本地知识检索**
   - 用户问及 LangChain / LangGraph / 项目本地资料 / "项目里有什么" →
     先调用 `search_docs` 查本地知识库，**必须基于检索结果回答**，不要凭空发挥。
3. **演示工具**
   - 用户明确要求做一次慢速查询 → 调用 `slow_lookup`。

# 兜底流程（重要）
- 如果你的内置知识能直接回答，优先直接回答。
- 如果你不确定或问题超出内置知识：
  1. 先尝试 `search_docs` 查本地知识库。
  2. 若本地知识库也没有结果，明确告诉用户"这个我目前查不到"，并建议他补充资料或换个问法。
- 不要凭空编造 API、配置项、代码细节。
"""


def build_agent():
    """组装并返回配置好的 LangChain agent。

    中间件按列表顺序串联执行：
      1. ModelCallLimit —— 防止 agent 死循环
      2. HumanInTheLoop —— slow_lookup 工具调用前暂停等人审批
      3. Summarization  —— token 接近上限时自动摘要老消息

    工具列表：
      - preference_tools（save_user_preference / get_user_preference）
      - demo_tools（slow_lookup，用于演示流式）
      - search_docs（RAG 检索本地知识库）
    """
    # RAG：构造本地 embedding + 加载（不是创建）Chroma 索引
    embeddings = build_embeddings()
    vectorstore = load_vectorstore(embeddings)

    # 混合检索：dense (Chroma/HNSW) + sparse (BM25 over chunk text) → RRF 融合。
    # BM25 索引持久化在 data/bm25_index.pkl，启动时 SHA1 校验自动重建。
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    bm25_retriever = load_or_build_bm25_retriever(vectorstore, k=8)
    retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        # weights / c / top_k 走 hybrid_retriever.py 的模块默认值
    )
    rag_tool = make_search_docs_tool(retriever)

    return create_agent(
        model=build_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=preference_tools + demo_tools + [rag_tool],
        middleware=[
            make_model_call_limit_middleware(),
            make_human_in_the_loop_middleware(),
            make_summarization_middleware(),
        ],
        checkpointer=create_checkpointer(),
        store=create_store(),
    )