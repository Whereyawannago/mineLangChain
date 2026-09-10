"""agent.context —— 上下文管理（持久化 + 身份隔离）

非中间件类的上下文管理机制：

- short_term: 短期记忆 —— checkpointer（默认 SqliteSaver，跨进程存活）
- long_term:  长期记忆 —— Store（默认 SqliteStore）+ preference 工具，按 user_id 隔离
- identity:   请求身份 —— UserContext / thread_id 生成与解析
- backends:   后端选择 —— sqlite（默认）/ memory，DB 路径

注意：消息摘要（SummarizationMiddleware）属于中间件，移到了 agent.middleware 子包。

典型用法：
    from agent.context import UserContext, new_thread_id

    ctx = UserContext(user_id="alice", thread_id=new_thread_id("alice", "sess-1"))
    agent.invoke({"messages": [...]},
                 config={"configurable": {"thread_id": ctx.thread_id}},
                 context=ctx)
"""

from .backends import (
    backend_name,
    checkpoint_db_path,
    open_sqlite,
    store_db_path,
    use_memory_backend,
)
from .identity import (
    DEFAULT_USER_ID,
    UserContext,
    new_thread_id,
    user_id_from_context,
    user_id_from_runtime,
)
from .long_term import (
    create_store,
    get_user_preference,
    pref_namespace,
    preference_tools,
    save_user_preference,
)
from .short_term import create_checkpointer

__all__ = [
    # 身份
    "DEFAULT_USER_ID",
    "UserContext",
    "new_thread_id",
    "user_id_from_context",
    "user_id_from_runtime",
    # 后端 / 路径
    "backend_name",
    "checkpoint_db_path",
    "store_db_path",
    "open_sqlite",
    "use_memory_backend",
    # 短期记忆
    "create_checkpointer",
    # 长期记忆
    "create_store",
    "get_user_preference",
    "save_user_preference",
    "preference_tools",
    "pref_namespace",
]
