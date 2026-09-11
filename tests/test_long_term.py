"""长期记忆（偏好）的写入约束 + 过期回收测试。

不调模型、不碰磁盘（conftest 已把后端切到 memory）：
  直接经工具的 ``.func``（原始函数）调用，用假 runtime 注入 store + 身份。

覆盖防线 1（归一化 / 配额 / 值截断 / 删除 / 列举）与防线 2（TTL 过期 + purge）。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from agent.context import (
    UserContext,
    create_store,
    delete_user_preference,
    get_user_preference,
    list_user_preferences,
    pref_namespace,
    purge_expired_preferences,
    save_user_preference,
)
from agent.context.long_term import _normalize_key, _value_expired


def make_runtime(store, user_id: str = "alice"):
    """假 runtime：提供 store + 身份上下文（user_id_from_runtime 从这里读）。"""
    return SimpleNamespace(store=store, context=UserContext(user_id=user_id))


# ─────────────────────────── 防线 1：写入约束 ───────────────────────────


def test_normalize_key() -> None:
    """键名归一化：去空白 + 小写，压掉语义重复。"""
    assert _normalize_key("Name") == "name"
    assert _normalize_key("  Name  ") == "name"
    assert _normalize_key("LANGUAGE") == "language"
    # 超长截断到 _MAX_KEY_CHARS
    assert len(_normalize_key("k" * 100)) == 64


def test_save_and_get_roundtrip() -> None:
    """保存 → 读取回环；键被归一化，值里带 updated_at 时间戳。"""
    store = create_store()
    rt = make_runtime(store)

    out = save_user_preference.func("Name", "Alice", runtime=rt)
    assert "已保存" in out

    item = store.get(pref_namespace("alice"), "name")
    assert item is not None
    assert item.value["value"] == "Alice"
    assert "updated_at" in item.value

    # 用不同大小写的键读，仍能命中（归一化）
    assert get_user_preference.func("NAME", runtime=rt) == "Alice"


def test_get_missing_key() -> None:
    store = create_store()
    rt = make_runtime(store)
    assert "未找到" in get_user_preference.func("missing", runtime=rt)


def test_quota_limits_new_keys(monkeypatch) -> None:
    """满额后拒绝新建键，但覆盖已有键不受限（合并语义）。"""
    monkeypatch.setenv("AGENT_PREF_MAX_KEYS", "2")
    store = create_store()
    rt = make_runtime(store)

    assert "已保存" in save_user_preference.func("k1", "v1", runtime=rt)
    assert "已保存" in save_user_preference.func("k2", "v2", runtime=rt)

    # 第 3 个新键被拒
    out = save_user_preference.func("k3", "v3", runtime=rt)
    assert "上限" in out
    assert store.get(pref_namespace("alice"), "k3") is None

    # 覆盖已有键仍允许
    assert "已保存" in save_user_preference.func("k1", "v1-updated", runtime=rt)
    assert store.get(pref_namespace("alice"), "k1").value["value"] == "v1-updated"


def test_value_truncation(monkeypatch) -> None:
    """值超长被截断到上限。"""
    monkeypatch.setenv("AGENT_PREF_MAX_VALUE_CHARS", "5")
    store = create_store()
    rt = make_runtime(store)

    save_user_preference.func("name", "Alice超长的值", runtime=rt)
    assert store.get(pref_namespace("alice"), "name").value["value"] == "Alice"


def test_empty_key_or_value_not_saved() -> None:
    store = create_store()
    rt = make_runtime(store)
    assert "未保存" in save_user_preference.func("   ", "x", runtime=rt)
    assert "未保存" in save_user_preference.func("k", "   ", runtime=rt)
    assert list(store.search(pref_namespace("alice"))) == []


def test_delete_user_preference() -> None:
    store = create_store()
    rt = make_runtime(store)
    save_user_preference.func("name", "Alice", runtime=rt)

    out = delete_user_preference.func("Name", runtime=rt)  # 归一化命中
    assert "已删除" in out
    assert store.get(pref_namespace("alice"), "name") is None

    out2 = delete_user_preference.func("name", runtime=rt)
    assert "无需删除" in out2


def test_list_user_preferences() -> None:
    store = create_store()
    rt = make_runtime(store)
    save_user_preference.func("name", "Alice", runtime=rt)
    save_user_preference.func("language", "中文", runtime=rt)

    out = list_user_preferences.func(runtime=rt)
    assert "name=Alice" in out
    assert "language=中文" in out

    empty = list_user_preferences.func(runtime=make_runtime(create_store(), "bob"))
    assert "没有任何" in empty


# ─────────────────────────── 防线 2：过期回收 ───────────────────────────


def test_value_expired(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PREF_TTL_DAYS", "1")
    ttl = 86400.0
    assert _value_expired({"updated_at": 100}, 100 + ttl + 1) is True
    assert _value_expired({"updated_at": 100}, 100 + ttl - 1) is False
    # 旧数据没有 updated_at → 视为不过期（安全）
    assert _value_expired({}, 10**9) is False

    # TTL <= 0 → 永不过期
    monkeypatch.setenv("AGENT_PREF_TTL_DAYS", "0")
    assert _value_expired({"updated_at": 100}, 10**9) is False


def test_get_expired_self_heals(monkeypatch) -> None:
    """读取已过期偏好时返回"未找到"并顺手删除（自愈）。"""
    monkeypatch.setenv("AGENT_PREF_TTL_DAYS", "1")
    store = create_store()
    rt = make_runtime(store)
    save_user_preference.func("name", "Alice", runtime=rt)

    # 快进 2 天
    future = time.time() + 2 * 86400
    monkeypatch.setattr("agent.context.long_term._now_ts", lambda: future)

    assert "未找到" in get_user_preference.func("name", runtime=rt)
    assert store.get(pref_namespace("alice"), "name") is None


def test_purge_expired_preferences_cross_user(monkeypatch) -> None:
    """purge 只删过期项、跨用户生效、返回删除数。"""
    monkeypatch.setenv("AGENT_PREF_TTL_DAYS", "1")
    store = create_store()
    base = int(time.time())

    store.put(pref_namespace("alice"), "old", {"value": "x", "updated_at": base - 100000})
    store.put(pref_namespace("bob"), "old", {"value": "y", "updated_at": base - 100000})
    store.put(pref_namespace("alice"), "fresh", {"value": "z", "updated_at": base})

    deleted = purge_expired_preferences(store, now_ts=base)
    assert deleted == 2
    assert store.get(pref_namespace("alice"), "old") is None
    assert store.get(pref_namespace("bob"), "old") is None
    assert store.get(pref_namespace("alice"), "fresh") is not None


def test_purge_noop_when_ttl_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PREF_TTL_DAYS", "0")
    store = create_store()
    store.put(pref_namespace("alice"), "old", {"value": "x", "updated_at": 0})
    assert purge_expired_preferences(store, now_ts=10**9) == 0
    assert store.get(pref_namespace("alice"), "old") is not None
