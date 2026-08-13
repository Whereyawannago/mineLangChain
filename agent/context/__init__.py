"""agent.context —— 上下文管理三大机制

按 LangChain 官方文档（docs.langchain.com/context-engineering）的分类：

- short_term: 短期记忆 —— 基于 checkpointer 的多轮对话（同 thread_id 跨轮保留）
- long_term:  长期记忆 —— 基于 Store 的跨会话持久化（通过工具读写）
- trim:       消息裁剪 —— @before_model 中间件按 token 阈值裁剪历史消息
"""

from .long_term import (
    USER_ID,
    create_store,
    get_user_preference,
    preference_tools,
    save_user_preference,
)
from .short_term import create_checkpointer
from .trim import MAX_CONTEXT_TOKENS, make_trim_middleware

__all__ = [
    "USER_ID",
    "MAX_CONTEXT_TOKENS",
    "create_checkpointer",
    "create_store",
    "get_user_preference",
    "preference_tools",
    "save_user_preference",
    "make_trim_middleware",
]