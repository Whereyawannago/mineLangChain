"""agent.context —— 上下文管理

非中间件类的上下文管理机制：

- short_term: 短期记忆 —— 基于 checkpointer 的多轮对话（同 thread_id 跨轮保留）
- long_term:  长期记忆 —— 基于 Store 的跨会话持久化（通过工具读写）

注意：消息摘要（SummarizationMiddleware）属于中间件，移到了 agent.middleware 子包。
"""

from .long_term import (
    USER_ID,
    create_store,
    get_user_preference,
    preference_tools,
    save_user_preference,
)
from .short_term import create_checkpointer

__all__ = [
    "USER_ID",
    "create_checkpointer",
    "create_store",
    "get_user_preference",
    "preference_tools",
    "save_user_preference",
]