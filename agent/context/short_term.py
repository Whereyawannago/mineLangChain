"""短期记忆（Short-term memory）

基于 LangGraph 的 checkpointer 机制。同一个 thread_id 内的多轮消息会自动保留，
由 LangChain 1.x 的 create_agent 在 invoke 时按 config={"configurable": {"thread_id": ...}} 自动加载。

官方文档：
    https://docs.langchain.com/oss/python/langchain/short-term-memory
"""

from langgraph.checkpoint.memory import InMemorySaver


def create_checkpointer() -> InMemorySaver:
    """构造一个内存 checkpointer。

    生产环境通常换成 PostgresSaver / SqliteSaver 等持久化实现。
    """
    return InMemorySaver()