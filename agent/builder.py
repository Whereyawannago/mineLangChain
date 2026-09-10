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

from typing import TypeVar

from langchain.agents import create_agent
from pydantic import BaseModel

from .context import (
    UserContext,
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
from .structured import make_response_format
from .tools import demo_tools

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

# 进程内复用 checkpointer / store —— 避免每次 build_* 都新开一条 sqlite 连接，
# 也保证同一进程内所有 agent 共用同一份持久化状态（多用户服务常见做法）。
_CHECKPOINTER = None
_STORE = None


def _get_checkpointer():
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = create_checkpointer()
    return _CHECKPOINTER


def _get_store():
    global _STORE
    if _STORE is None:
        _STORE = create_store()
    return _STORE

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


# RAG 每路 retriever 的候选数（dense / sparse 各自取 k 个，再由 RRF 融合成 top_k）
_RETRIEVER_K = 8


def _assemble_rag_tool():
    """构造 search_docs 工具：混合检索（向量 + BM25）→ RRF 融合。

    两个 build_* 入口共用这一套装配，避免逻辑重复：
    加载 Chroma → dense retriever + BM25 retriever → HybridRetriever → @tool。

    说明：
      - BM25 索引持久化在 data/bm25_index.pkl，启动时按 Chroma 内容 SHA1 校验自动重建。
      - weights / c / top_k 走 hybrid_retriever.py 的模块默认值。
    """
    embeddings = build_embeddings()
    vectorstore = load_vectorstore(embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": _RETRIEVER_K})
    bm25_retriever = load_or_build_bm25_retriever(vectorstore, k=_RETRIEVER_K)
    retriever = HybridRetriever(retrievers=[vector_retriever, bm25_retriever])
    return make_search_docs_tool(retriever)


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
    # RAG：dense (Chroma) + sparse (BM25) → RRF 融合，包成 search_docs 工具
    rag_tool = _assemble_rag_tool()

    return create_agent(
        model=build_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=preference_tools + demo_tools + [rag_tool],
        middleware=[
            make_model_call_limit_middleware(),
            make_human_in_the_loop_middleware(),
            make_summarization_middleware(),
        ],
        checkpointer=_get_checkpointer(),
        store=_get_store(),
        # 身份上下文：invoke(..., context=UserContext(user_id=...)) 时供 preference 工具读取
        context_schema=UserContext,
    )


def build_structured_agent(schema: type[_SchemaT] | None = None, *, include_rag: bool = True):
    """组装一个「结构化输出」版本的 agent。

    区别于 `build_agent()`：
      - 通过 `response_format=ToolStrategy(schema)` 把模型回复强制收敛到 Pydantic schema。
      - 调用后 `result["structured_response"]` 直接是 schema 实例（不是字符串）。
      - **不挂 slow_lookup**（结构化输出场景一般是后端业务，不该慢）。

    Args:
        schema:  Pydantic BaseModel 子类。None 时用 `ChatReply`（最小可用形态）。
        include_rag: 是否挂 search_docs 工具。False 时只走内置知识 + 记忆。

    Returns:
        配置好的 agent，调用方式与 `build_agent()` 相同，但 invoke 返回值
        多一个 `structured_response` 键。

    示例：
        from agent.structured import WeatherReport
        agent = build_structured_agent(WeatherReport, include_rag=False)
        result = agent.invoke({"messages": [{"role": "user", "content": "北京天气？"}]})
        report = result["structured_response"]
        print(report.city, report.temperature)
    """
    from .structured import ChatReply, make_response_format

    schema = schema or ChatReply
    tools: list = list(preference_tools)
    if include_rag:
        # 与 build_agent() 共享同一套混合检索装配，保持单数据源
        tools.append(_assemble_rag_tool())

    return create_agent(
        model=build_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        middleware=[
            make_model_call_limit_middleware(),
            # 注意：结构化输出场景默认不开 HITL —— ToolStrategy 已经把输出结构化，
            # 人为审批反而打断业务集成。要开可手动加。
            # make_human_in_the_loop_middleware(),
            make_summarization_middleware(),
        ],
        # 结构化输出用 `response_format=...` 而不是消息里的 tool_call
        response_format=make_response_format(schema),
        checkpointer=_get_checkpointer(),
        store=_get_store(),
        context_schema=UserContext,
    )