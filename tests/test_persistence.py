"""sqlite 记忆后端的持久化 / 隔离测试。

不调模型：直接用 create_checkpointer / create_store 读写，验证
  1. 数据真的落到 sqlite（换一个 store 实例仍读得到）；
  2. 长期记忆按 user_id 隔离；
  3. 关闭再重开连接（模拟重启）后数据还在。

这些用例显式把后端切回 sqlite 并指向 tmp_path，绕开 conftest 的 memory 默认值。
"""

from __future__ import annotations

import pytest

from agent.context import (
    create_checkpointer,
    create_store,
    pref_namespace,
    store_db_path,
)


@pytest.fixture
def sqlite_env(tmp_path, monkeypatch):
    """把记忆后端切到 sqlite，DB 文件落在 tmp_path。"""
    monkeypatch.setenv("AGENT_MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("AGENT_MEMORY_STORE_DB", str(tmp_path / "store.sqlite"))
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "checkpoints.sqlite"))
    yield tmp_path


def test_store_uses_sqlite_when_backend_sqlite(sqlite_env) -> None:
    from langgraph.store.sqlite import SqliteStore

    store = create_store()
    assert isinstance(store, SqliteStore)
    assert store_db_path().exists()


def test_checkpointer_uses_sqlite_when_backend_sqlite(sqlite_env) -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = create_checkpointer()
    assert isinstance(saver, SqliteSaver)


def test_store_survives_new_connection(sqlite_env) -> None:
    """写入后用**新连接**（模拟重启）读回 —— 证明数据真落在磁盘而非内存。"""
    store1 = create_store()
    store1.put(pref_namespace("alice"), "name", {"value": "Alice"})

    store2 = create_store()  # 全新连接
    item = store2.get(pref_namespace("alice"), "name")
    assert item is not None
    assert item.value["value"] == "Alice"


def test_store_isolated_per_user(sqlite_env) -> None:
    """alice 的偏好不能出现在 bob 的命名空间里。"""
    store = create_store()
    store.put(pref_namespace("alice"), "name", {"value": "Alice"})

    assert store.get(pref_namespace("bob"), "name") is None
    assert store.get(pref_namespace("alice"), "name").value["value"] == "Alice"


def test_no_database_locked_when_both_open(monkeypatch, tmp_path) -> None:
    """checkpointer 与 store 同时打开不应报 database is locked。

    回归用例：早期两者共用同一个 sqlite 文件，setup() 会撞锁。
    现在各用独立文件（checkpoints.sqlite / store.sqlite）。
    """
    monkeypatch.setenv("AGENT_MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("AGENT_MEMORY_DB", str(tmp_path / "cp.sqlite"))
    monkeypatch.setenv("AGENT_MEMORY_STORE_DB", str(tmp_path / "st.sqlite"))

    saver = create_checkpointer()
    store = create_store()
    assert saver is not None and store is not None
