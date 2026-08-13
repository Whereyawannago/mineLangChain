"""长期记忆（Long-term memory）

基于 LangGraph 的 Store 机制。数据按 (namespace, key) 组织，跨会话持久化。
Agent 通过工具读写：save_user_preference 写入，get_user_preference 读取。

官方文档：
    https://docs.langchain.com/oss/python/langchain/long-term-memory
"""

from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

# 演示用的 user_id 和存储命名空间
USER_ID = "demo-user"
PREF_NAMESPACE = ("user_preferences", USER_ID)


@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """保存一条用户偏好到长期记忆。

    当用户告诉你他的偏好时（比如名字、语言、称呼），调用此工具保存。
    之后任何会话都可以用 get_user_preference 读回。

    Args:
        key: 偏好的键名，例如 "name"、"language"。
        value: 偏好的值，例如 "Alice"、"中文"。
    """
    runtime.store.put(PREF_NAMESPACE, key, {"value": value})
    return f"已保存偏好：{key}={value}"


@tool
def get_user_preference(key: str, runtime: ToolRuntime) -> str:
    """从长期记忆读取一条用户偏好。

    当用户问起他之前的偏好时，调用此工具查询。

    Args:
        key: 要查询的偏好键名。
    """
    item = runtime.store.get(PREF_NAMESPACE, key)
    if item is None:
        return f"未找到键 {key} 的偏好"
    return str(item.value.get("value", ""))


# 工具列表，方便 create_agent 直接传入
preference_tools = [save_user_preference, get_user_preference]


def create_store() -> InMemoryStore:
    """构造一个内存 Store。

    生产环境通常换成 PostgresStore 等持久化实现。
    """
    return InMemoryStore()