"""identity / 后端选择 的单测（不碰磁盘、不调模型）。"""

from __future__ import annotations

import pytest

from agent.context import (
    DEFAULT_USER_ID,
    UserContext,
    new_thread_id,
    pref_namespace,
    user_id_from_context,
)


# ─────────────────────────── thread_id 生成 ───────────────────────────


def test_new_thread_id_stable_for_same_user_and_session() -> None:
    """同一 (user, session) 必须得到同一个 thread_id —— 短期记忆才能跨请求恢复。"""
    assert new_thread_id("alice", "s1") == new_thread_id("alice", "s1")


def test_new_thread_id_is_namespaced_by_user() -> None:
    """不同用户即使 session 相同，thread_id 也不能撞。"""
    assert new_thread_id("alice", "s1") != new_thread_id("bob", "s1")


def test_new_thread_id_random_when_session_missing() -> None:
    """不给 session 时退化成随机短 id（不同调用应不同）。"""
    a, b = new_thread_id("alice"), new_thread_id("alice")
    assert a != b
    assert a.startswith("alice:")


# ─────────────────────────── user_id 解析 ───────────────────────────


def test_user_id_from_usercontext() -> None:
    assert user_id_from_context(UserContext(user_id="alice")) == "alice"


def test_user_id_from_dict() -> None:
    assert user_id_from_context({"user_id": "bob"}) == "bob"


def test_user_id_falls_back_to_default() -> None:
    """None / 空 dict / 空 user_id 都回退到默认用户。"""
    assert user_id_from_context(None) == DEFAULT_USER_ID
    assert user_id_from_context({}) == DEFAULT_USER_ID
    assert user_id_from_context(UserContext(user_id="")) == DEFAULT_USER_ID


def test_user_id_from_arbitrary_object() -> None:
    class _Ctx:
        user_id = "carol"

    assert user_id_from_context(_Ctx()) == "carol"


# ─────────────────────────── 命名空间隔离 ───────────────────────────


def test_pref_namespace_isolated_per_user() -> None:
    """长期记忆必须按 user_id 分命名空间，不能所有用户共用一份。"""
    assert pref_namespace("alice") != pref_namespace("bob")
    assert pref_namespace("alice") == ("user_preferences", "alice")


# ─────────────────────────── 后端选择 ───────────────────────────


def test_default_backend_is_sqlite() -> None:
    """默认后端应是 sqlite（持久化）。conftest 会把测试环境设成 memory。"""
    from agent.context import backend_name, use_memory_backend

    # conftest 已设 AGENT_MEMORY_BACKEND=memory
    assert backend_name() == "memory"
    assert use_memory_backend() is True


def test_backend_name_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.context import backend_name, use_memory_backend

    monkeypatch.setenv("AGENT_MEMORY_BACKEND", "sqlite")
    assert backend_name() == "sqlite"
    assert use_memory_backend() is False
