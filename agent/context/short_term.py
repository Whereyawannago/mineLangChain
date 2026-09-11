"""短期记忆（Short-term memory）—— checkpointer。

同一 thread_id 内的多轮消息由 create_agent 在 invoke/stream 时按
``config={"configurable": {"thread_id": ...}}`` 自动加载与保存。

默认后端是 **SqliteSaver**：写入 ``data/memory/checkpoints.sqlite``，
进程重启后同一 thread_id 的对话仍在（跨进程存活）。

thread_id 必须来自请求身份，不能每次会话随机 uuid —— 否则重启就"失忆"。
生成/解析见 ``identity.new_thread_id``（形如 ``<user_id>:<session>``）。

环境变量：
    AGENT_MEMORY_BACKEND=memory   → 退回 InMemorySaver（测试 / CI）
    AGENT_MEMORY_DB=<path>        → 自定义 checkpointer 的 sqlite 文件
                                    （与 store 的 AGENT_MEMORY_STORE_DB 分开，避锁争用）

生产多实例：换 ``langgraph.checkpoint.postgres.PostgresSaver``（见 README「生产化」）。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .backends import checkpoint_db_path, open_sqlite, use_memory_backend


def create_checkpointer() -> BaseCheckpointSaver:
    """构造 checkpointer：默认 SqliteSaver（持久化），可用环境变量退回内存。

    Returns:
        已 ``setup()`` 过的 SqliteSaver，或 InMemorySaver。
    """
    if use_memory_backend():
        return InMemorySaver()

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = open_sqlite(checkpoint_db_path())
    saver = SqliteSaver(conn)
    saver.setup()  # 首次运行建表；重复调用幂等
    return saver
