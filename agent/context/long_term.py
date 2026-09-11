"""长期记忆（Long-term memory）—— Store。

数据按 ``(namespace, key)`` 组织，跨会话、跨进程持久化。Agent 通过工具读写：
``save_user_preference`` 写入，``get_user_preference`` 读取。

两处隔离（这是相对早期版本的关键修正）：
  1. **命名空间按 user_id 隔离**：``("user_preferences", <user_id>)``。
     早期用模块级常量 ``USER_ID = "demo-user"``，所有用户共享一份偏好——那是 bug 不是特性。
     user_id 现在从请求上下文取（``context.identity.user_id_from_runtime``）。
  2. **后端默认 SqliteStore**：写 ``data/memory/store.sqlite``，进程重启仍在。
     早期 InMemoryStore 只在同进程内跨会话有效，docstring 宣称的"跨会话持久化"并不成立。

防无限增长（写入约束 + 过期回收）：
  - **键/值有界**：键名归一化（小写/去空白/截断），值有长度上限；
  - **每用户配额**：偏好条数有上限，满额后引导模型合并/删除而非无脑新建键；
  - **过期回收**：每条偏好带 ``updated_at`` 时间戳，超过 TTL 的条目读取时自愈删除，
    并可经 ``purge_expired_preferences`` 定时全量清理；
  - **删除/列举工具**：``delete_user_preference`` / ``list_user_preferences`` 补齐
    "读/写/删/列"闭环，满足"忘掉某条"与 GDPR 遗忘权。

环境变量：AGENT_MEMORY_BACKEND=memory 退回 InMemoryStore（测试/CI）；
          AGENT_MEMORY_STORE_DB=<path> 自定义 store 的 sqlite 文件
          （与 checkpointer 的 AGENT_MEMORY_DB 分开，避免跨连接写锁冲突）；
          AGENT_PREF_MAX_KEYS=<n>        每用户偏好条数上限（默认 50）
          AGENT_PREF_MAX_VALUE_CHARS=<n> 单条值长度上限（默认 500）
          AGENT_PREF_TTL_DAYS=<n>        偏好过期天数（默认 90；<=0 不过期）

生产多实例：换 ``langgraph.store.postgres.PostgresStore``（见 README「生产化」）。
"""

from __future__ import annotations

import os
import time

from langchain.tools import ToolRuntime, tool
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from .backends import open_sqlite, store_db_path, use_memory_backend
from .identity import user_id_from_runtime


def pref_namespace(user_id: str) -> tuple[str, str]:
    """某个用户的偏好命名空间。所有读写都必须经这里，保证按用户隔离。"""
    return ("user_preferences", user_id)


# ────────────────────────────── 配额 / 约束 ──────────────────────────────
#
# 这里用「调用期读 env」而不是模块级常量：与 identity.user_allowed_dirs() 同理，
# 让 monkeypatch.setenv / 配置中心热更新能生效，测试里也能把上限调小来测配额。

#: 键名长度上限（归一化后截断到此长度）
_MAX_KEY_CHARS = 64


def _max_pref_keys() -> int:
    """每用户偏好条数上限（配额）。"""
    return int(os.getenv("AGENT_PREF_MAX_KEYS", "50"))


def _max_value_chars() -> int:
    """单条偏好值长度上限。"""
    return int(os.getenv("AGENT_PREF_MAX_VALUE_CHARS", "500"))


def _pref_ttl_seconds() -> float:
    """偏好过期时长（秒）。<=0 表示不过期。"""
    return int(os.getenv("AGENT_PREF_TTL_DAYS", "90")) * 86400.0


def _now_ts() -> float:
    """当前时间戳。单独抽出来方便测试 monkeypatch（避免动全局 time.time）。"""
    return time.time()


def _normalize_key(key: str) -> str:
    """键名归一化：去空白 + 小写 + 截断，压掉 "Name"/"name" 这类语义重复。"""
    return key.strip().lower()[:_MAX_KEY_CHARS]


def _value_expired(value: dict, now_ts: float) -> bool:
    """判断一条偏好是否已过期。旧数据没有 updated_at 视为不过期（安全）。"""
    ttl = _pref_ttl_seconds()
    if ttl <= 0:
        return False
    ts = (value or {}).get("updated_at")
    return bool(ts) and (now_ts - float(ts)) > ttl


@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """保存一条用户偏好到长期记忆。

    当用户告诉你他的偏好时（比如名字、语言、称呼），调用此工具保存。
    之后任何会话都可以用 get_user_preference 读回。

    注意：保存前先调用 list_user_preferences 查重，语义相同的偏好复用旧键；
    偏好条数有上限，满额时请合并或删除旧偏好，而不是无脑新建键。

    Args:
        key: 偏好的键名，例如 "name"、"language"。
        value: 偏好的值，例如 "Alice"、"中文"。
    """
    user_id = user_id_from_runtime(runtime)
    key = _normalize_key(key)
    value = (value or "").strip()[:_max_value_chars()]
    if not key or not value:
        return "偏好键或值为空，未保存。"

    ns = pref_namespace(user_id)
    # 配额：只对新键计数；覆盖已有键不受限（合并语义）。
    # search 默认 limit=10，必须显式放大，否则满额计数会错。
    existing = list(runtime.store.search(ns, limit=_max_pref_keys() + 1))
    is_new_key = all(it.key != key for it in existing)
    if is_new_key and len(existing) >= _max_pref_keys():
        return (
            f"已达到每用户偏好上限（{_max_pref_keys()} 条）。"
            "请先调用 list_user_preferences 查看，合并重复项或调用 delete_user_preference 删除旧偏好。"
        )

    runtime.store.put(ns, key, {"value": value, "updated_at": int(_now_ts())})
    return f"已保存偏好：{key}={value}"


@tool
def get_user_preference(key: str, runtime: ToolRuntime) -> str:
    """从长期记忆读取一条用户偏好。

    当用户问起他之前的偏好时，调用此工具查询。

    Args:
        key: 要查询的偏好键名。
    """
    user_id = user_id_from_runtime(runtime)
    key = _normalize_key(key)
    ns = pref_namespace(user_id)
    item = runtime.store.get(ns, key)
    if item is None:
        return f"未找到键 {key} 的偏好"
    if _value_expired(item.value, _now_ts()):
        # 过期自愈：顺手删掉，减少对定时 janitor 的依赖
        runtime.store.delete(ns, key)
        return f"未找到键 {key} 的偏好"
    return str(item.value.get("value", ""))


@tool
def list_user_preferences(runtime: ToolRuntime) -> str:
    """列出当前用户已保存的所有偏好。

    用于保存前查重、合并、删除。没有偏好时返回提示。
    """
    user_id = user_id_from_runtime(runtime)
    items = list(runtime.store.search(pref_namespace(user_id), limit=200))
    if not items:
        return "当前没有任何已保存的偏好"
    return "\n".join(f"{it.key}={it.value.get('value', '')}" for it in items)


@tool
def delete_user_preference(key: str, runtime: ToolRuntime) -> str:
    """删除一条用户偏好。

    当用户明确表示"忘记/不要记住/删掉"某条偏好时调用，真正从长期记忆移除。

    Args:
        key: 要删除的偏好键名。
    """
    user_id = user_id_from_runtime(runtime)
    ns = pref_namespace(user_id)
    key = _normalize_key(key)
    if runtime.store.get(ns, key) is None:
        return f"未找到键 {key}，无需删除"
    runtime.store.delete(ns, key)
    return f"已删除偏好：{key}"


def purge_expired_preferences(store: BaseStore, *, now_ts: float | None = None) -> int:
    """删除所有已过期的偏好，返回删除条数。由定时任务 / 后台线程调用。

    分页扫描全量命名空间（search 默认 limit=10，必须显式翻页），跨所有用户生效。
    TTL <= 0（未启用过期）时直接返回 0。
    """
    if _pref_ttl_seconds() <= 0:
        return 0
    now_ts = _now_ts() if now_ts is None else now_ts
    deleted = 0
    offset, limit = 0, 200
    while True:
        page = list(store.search(("user_preferences",), limit=limit, offset=offset))
        if not page:
            break
        for item in page:
            if _value_expired(item.value, now_ts):
                store.delete(item.namespace, item.key)
                deleted += 1
        if len(page) < limit:
            break
        offset += limit
    return deleted


# 工具列表，方便 create_agent 直接传入
preference_tools = [
    save_user_preference,
    get_user_preference,
    list_user_preferences,
    delete_user_preference,
]


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
