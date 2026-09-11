"""长期记忆（Long-term memory）—— Store。

数据按 ``(namespace, key)`` 组织，跨会话、跨进程持久化。Agent 通过工具读写：
``save_user_preference`` 写入，``get_user_preference`` 读取。

两处隔离（这是相对早期版本的关键修正）：
  1. **命名空间按 user_id 隔离**：``("user_preferences", <user_id>)``。
     早期用模块级常量 ``USER_ID = "demo-user"``，所有用户共享一份偏好——那是 bug 不是特性。
     user_id 现在从请求上下文取（``context.identity.user_id_from_runtime``）。
  2. **后端默认 SqliteStore**：写 ``data/memory/store.sqlite``，进程重启仍在。
     早期 InMemoryStore 只在同进程内跨会话有效，docstring 宣称的"跨会话持久化"并不成立。

环境变量：AGENT_MEMORY_BACKEND=memory 退回 InMemoryStore（测试/CI）；
          AGENT_MEMORY_STORE_DB=<path> 自定义 store 的 sqlite 文件
          （与 checkpointer 的 AGENT_MEMORY_DB 分开，避免跨连接写锁冲突）。

生产多实例：换 ``langgraph.store.postgres.PostgresStore``（见 README「生产化」）。
"""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from .backends import open_sqlite, store_db_path, use_memory_backend
from .identity import user_id_from_runtime


def pref_namespace(user_id: str) -> tuple[str, str]:
    """某个用户的偏好命名空间。所有读写都必须经这里，保证按用户隔离。"""
    return ("user_preferences", user_id)


@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """保存一条用户偏好到长期记忆。

    当用户告诉你他的偏好时（比如名字、语言、称呼），调用此工具保存。
    之后任何会话都可以用 get_user_preference 读回。

    Args:
        key: 偏好的键名，例如 "name"、"language"。
        value: 偏好的值，例如 "Alice"、"中文"。
    """
    user_id = user_id_from_runtime(runtime)
    runtime.store.put(pref_namespace(user_id), key, {"value": value})
    return f"已保存偏好：{key}={value}"


@tool
def get_user_preference(key: str, runtime: ToolRuntime) -> str:
    """从长期记忆读取一条用户偏好。

    当用户问起他之前的偏好时，调用此工具查询。

    Args:
        key: 要查询的偏好键名。
    """
    user_id = user_id_from_runtime(runtime)
    item = runtime.store.get(pref_namespace(user_id), key)
    if item is None:
        return f"未找到键 {key} 的偏好"
    return str(item.value.get("value", ""))


# 工具列表，方便 create_agent 直接传入
preference_tools = [save_user_preference, get_user_preference]


def create_store() -> BaseStore:
    """构造 Store：默认 SqliteStore（持久化），可用环境变量退回内存。

    Returns:
        已 ``setup()`` 过的 SqliteStore，或 InMemoryStore。
    """
    if use_memory_backend():
        return InMemoryStore()

    from langgraph.store.sqlite import SqliteStore

    conn = open_sqlite(store_db_path())
    store = SqliteStore(conn)
    store.setup()  # 首次运行建表；重复调用幂等
    return store
