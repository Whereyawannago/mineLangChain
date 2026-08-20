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
from .rag import build_embeddings, load_vectorstore, make_search_docs_tool
from .tools import demo_tools

SYSTEM_PROMPT = (
    "你是一个友好、简洁的中文助手。"
    "回答时先给一句话结论，再用要点展开。"
    "当用户告诉你他的偏好（如名字、语言、称呼），用 save_user_preference 工具保存。"
    "当用户问起他的偏好，用 get_user_preference 工具查询。"
    "当用户问及 LangChain / LangGraph / 项目本地资料，"
    "先用 search_docs 工具查本地知识库，再结合检索到的内容回答。"
    "如果用户要求做一次慢速查询或查询某些信息，可以用 slow_lookup 工具。"
    "如果不在你的知识范围内，直接告诉用户你无法回答。"
)


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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
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