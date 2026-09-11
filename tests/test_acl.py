"""多租户 RAG 隔离 (ACL) 测试。

覆盖：
  - 角色 → allowed_paths 解析；
  - user 白名单的惰性解析（环境变量改了立即生效）；
  - admin 放行全集；user 仅 BackEndNote 下的文档；
  - 反斜杠路径归一化；
  - ContextVar 隔离（同一进程内 set/reset）；
  - **fail-closed**：拿不到请求上下文时返回空集，显式 role=admin 才放行；
  - ACLRetriever 包装后的 invoke 在不同 context 下结果不同。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agent.context import (
    ROLE_ADMIN,
    ROLE_USER,
    UserContext,
    allowed_paths_for_role,
    reset_current_context,
    set_current_context,
)
from agent.context.identity import user_allowed_dirs
from agent.rag.acl import ACLRetriever, doc_visible_to_role, visible_docs


def _doc(source: str, content: str = "x") -> Document:
    return Document(page_content=content, metadata={"source": source})


@pytest.fixture(autouse=True)
def _clear_context():
    """每个用例都在「无请求上下文」的干净状态下开始，避免上一个用例的 ContextVar 泄漏。"""
    token = set_current_context(None)
    yield
    reset_current_context(token)


# ─────────────────────────── 角色 → allowed_paths ───────────────────────────


def test_admin_has_no_allowed_paths_filter() -> None:
    """admin 不在路径过滤名单里 —— 调用方把它视作「全集放行」。"""
    assert allowed_paths_for_role(ROLE_ADMIN) == ()


def test_user_default_allowed_is_obsidian_backend_dir() -> None:
    """user 角色默认只看 BackEndNote。"""
    paths = allowed_paths_for_role(ROLE_USER)
    assert len(paths) >= 1
    assert any("BackEndNote" in str(p) for p in paths)


def test_user_allowed_paths_are_absolute() -> None:
    """解析后必须是绝对路径，is_relative_to 才能稳定工作。"""
    for p in user_allowed_dirs():
        assert p.is_absolute()


def test_extra_allowed_paths_extend_role_whitelist(tmp_path: Path) -> None:
    """UserContext.extra_allowed_paths 应该叠加在角色默认白名单之上。"""
    base = allowed_paths_for_role(ROLE_USER)
    extended = allowed_paths_for_role(ROLE_USER, extra=(str(tmp_path),))
    assert extended == base + (tmp_path.resolve(),)


# ─────────────────── 白名单惰性解析（环境变量热更新） ───────────────────


def test_user_allowed_dirs_reads_env_on_every_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """白名单必须每次调用重新读 env，不能在 import 时固化成常量。

    早先的实现是模块级 ``USER_ALLOWED_DIRS = tuple(...)``：这里 monkeypatch 完全不生效，
    生产上也做不到不重启进程就换策略。
    """
    monkeypatch.setenv("AGENT_USER_ALLOWED_DIRS", str(tmp_path))
    assert user_allowed_dirs() == (tmp_path.resolve(),)

    other = tmp_path / "other"
    monkeypatch.setenv("AGENT_USER_ALLOWED_DIRS", str(other))
    assert user_allowed_dirs() == (other.resolve(),)


def test_user_allowed_dirs_accepts_multiple_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("AGENT_USER_ALLOWED_DIRS", f"{a},{b}")
    assert user_allowed_dirs() == (a.resolve(), b.resolve())


def test_user_allowed_dirs_relative_to_vault_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """不给绝对路径时，AGENT_USER_ALLOWED_DIRS_RELATIVE 解析到 vault 根下。"""
    monkeypatch.delenv("AGENT_USER_ALLOWED_DIRS", raising=False)
    monkeypatch.setenv("AGENT_OBSIDIAN_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_USER_ALLOWED_DIRS_RELATIVE", "TeamNotes,PublicNotes")

    assert user_allowed_dirs() == (
        (tmp_path / "TeamNotes").resolve(),
        (tmp_path / "PublicNotes").resolve(),
    )


def test_legacy_module_attr_is_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """旧写法 ``identity.USER_ALLOWED_DIRS`` 仍可用，且拿到的是当前 env 下的结果。"""
    import agent.context.identity as identity_mod

    monkeypatch.setenv("AGENT_USER_ALLOWED_DIRS", str(tmp_path))
    assert identity_mod.USER_ALLOWED_DIRS == (tmp_path.resolve(),)

    with pytest.raises(AttributeError):
        _ = identity_mod.NOT_A_REAL_ATTR


# ─────────────────────────── doc_visible_to_role ───────────────────────────


def test_admin_sees_anything_with_source() -> None:
    d = _doc(r"G:\ObsidianNote\LangChainNote\langChain\langChain.md")
    assert doc_visible_to_role(d, role=ROLE_ADMIN, allowed=()) is True


def test_user_only_sees_backend() -> None:
    backend = _doc(r"G:\ObsidianNote\BackEndNote\BackEnd\SpringBoot\SpringBoot学习指南.md")
    langchain = _doc(r"G:\ObsidianNote\LangChainNote\langChain\langChain.md")
    allowed = allowed_paths_for_role(ROLE_USER)
    assert doc_visible_to_role(backend, role=ROLE_USER, allowed=allowed) is True
    assert doc_visible_to_role(langchain, role=ROLE_USER, allowed=allowed) is False


def test_user_denied_when_no_allowed_paths() -> None:
    """user 角色若没有任何允许路径（极端情况），一律拒绝。"""
    d = _doc(r"G:\ObsidianNote\BackEndNote\BackEnd\anything.md")
    assert doc_visible_to_role(d, role=ROLE_USER, allowed=()) is False


def test_doc_without_source_is_denied_for_user() -> None:
    """没有 source 的 chunk 不能算「可见」—— 安全默认。"""
    assert doc_visible_to_role(_doc(""), role=ROLE_USER, allowed=(Path("G:/ObsidianNote"),)) is False
    assert doc_visible_to_role(Document(page_content="x"), role=ROLE_USER,
                               allowed=(Path("G:/ObsidianNote"),)) is False


def test_backslash_and_forward_slash_equivalent() -> None:
    """工具链里 source 可能是反斜杠或正斜杠，判定必须等价。"""
    allowed = (Path("G:/ObsidianNote/BackEndNote"),)
    a = _doc(r"G:\ObsidianNote\BackEndNote\foo.md")
    b = _doc("G:/ObsidianNote/BackEndNote/foo.md")
    assert doc_visible_to_role(a, role=ROLE_USER, allowed=allowed) == \
           doc_visible_to_role(b, role=ROLE_USER, allowed=allowed) is True


def test_parent_dir_traversal_does_not_escape_whitelist() -> None:
    """``..`` 不能用来跳出白名单根目录（resolve 之后前缀已经不匹配）。"""
    allowed = (Path("G:/ObsidianNote/BackEndNote").resolve(),)
    sneaky = _doc("G:/ObsidianNote/BackEndNote/../../LangChainNote/secret.md")
    assert doc_visible_to_role(sneaky, role=ROLE_USER, allowed=allowed) is False


# ─────────────────────────── visible_docs ───────────────────────────


def test_visible_docs_uses_context_when_no_explicit_role() -> None:
    """visible_docs 不传 role 时应从 ContextVar 取 —— 这是 retriever 调用路径。"""
    docs = [
        _doc(r"G:\ObsidianNote\BackEndNote\a.md", "be"),
        _doc(r"G:\ObsidianNote\LangChainNote\b.md", "lc"),
    ]
    token = set_current_context(UserContext(user_id="alice", role=ROLE_USER))
    try:
        out = visible_docs(docs)
    finally:
        reset_current_context(token)
    assert [d.page_content for d in out] == ["be"]


def test_visible_docs_fails_closed_without_context() -> None:
    """**没有请求上下文时返回空集**，不是原样放行。

    这是安全基线：只要有一条请求路径忘了 set_current_context，早先的
    fail-open 实现就会把整个知识库泄露出去，而且不报任何错。
    """
    docs = [
        _doc(r"G:\ObsidianNote\BackEndNote\a.md"),
        _doc(r"G:\ObsidianNote\LangChainNote\b.md"),
    ]
    assert visible_docs(docs) == []


def test_visible_docs_fails_closed_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    """fail-closed 必须留痕，否则排查「为什么检索空了」会毫无头绪。"""
    with caplog.at_level("WARNING", logger="agent.rag.acl"):
        assert visible_docs([_doc(r"G:\ObsidianNote\BackEndNote\a.md")]) == []
    assert any("fail-closed" in r.message for r in caplog.records)


def test_visible_docs_admin_role_explicitly_allows_everything() -> None:
    """离线场景（评测 / 灌库 / 管理后台）必须显式写 role=admin 才能放行全集。"""
    docs = [
        _doc(r"G:\ObsidianNote\BackEndNote\a.md", "be"),
        _doc(r"G:\ObsidianNote\LangChainNote\b.md", "lc"),
    ]
    assert [d.page_content for d in visible_docs(docs, role=ROLE_ADMIN)] == ["be", "lc"]


def test_visible_docs_explicit_allowed_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """显式传 (role, allowed) 时不再读环境变量白名单。"""
    monkeypatch.setenv("AGENT_USER_ALLOWED_DIRS", str(tmp_path / "other"))
    inside = tmp_path / "in"
    docs = [_doc(str(inside / "a.md"), "ok"), _doc(str(tmp_path / "out.md"), "no")]

    out = visible_docs(docs, role=ROLE_USER, allowed=(inside.resolve(),))

    assert [d.page_content for d in out] == ["ok"]


def test_visible_docs_consumes_any_iterable() -> None:
    """入参是 Iterable，生成器也要能正确过滤（内部先 list 化）。"""
    gen = (d for d in [_doc(r"G:\ObsidianNote\BackEndNote\a.md", "be")])
    token = set_current_context(UserContext(user_id="alice", role=ROLE_USER))
    try:
        assert [d.page_content for d in visible_docs(gen)] == ["be"]
    finally:
        reset_current_context(token)


# ─────────────────────────── ACLRetriever ───────────────────────────


class _FakeRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query: str, **kwargs):
        return list(self.docs)


def _fake(docs: list[Document]) -> _FakeRetriever:
    return _FakeRetriever(docs=docs)


def test_acl_retriever_is_a_real_base_retriever() -> None:
    """必须是 BaseRetriever 实例：LangChain 内部多处 isinstance 判定，鸭子类型会被判失败。"""
    acl = ACLRetriever(_fake([]))
    assert isinstance(acl, BaseRetriever)
    assert isinstance(acl.invoke, object)  # Runnable 接口可用


def test_acl_retriever_filters_by_role() -> None:
    acl = ACLRetriever(_fake([
        _doc(r"G:\ObsidianNote\BackEndNote\a.md", "be"),
        _doc(r"G:\ObsidianNote\LangChainNote\b.md", "lc"),
    ]))

    token = set_current_context(UserContext(user_id="admin1", role=ROLE_ADMIN))
    try:
        admin_out = acl.invoke("anything")
    finally:
        reset_current_context(token)
    assert {d.page_content for d in admin_out} == {"be", "lc"}

    token = set_current_context(UserContext(user_id="user1", role=ROLE_USER))
    try:
        user_out = acl.invoke("anything")
    finally:
        reset_current_context(token)
    assert {d.page_content for d in user_out} == {"be"}


def test_acl_retriever_honours_extra_allowed_paths(tmp_path: Path) -> None:
    """UserContext.extra_allowed_paths 能让某个用户临时多看一个目录。"""
    extra = tmp_path / "SharedVault"
    acl = ACLRetriever(_fake([
        _doc(str(extra / "note.md"), "shared"),
        _doc(r"G:\ObsidianNote\LangChainNote\b.md", "lc"),
    ]))

    token = set_current_context(
        UserContext(user_id="u2", role=ROLE_USER, extra_allowed_paths=(str(extra),))
    )
    try:
        out = acl.invoke("anything")
    finally:
        reset_current_context(token)
    assert {d.page_content for d in out} == {"shared"}


def test_acl_retriever_fails_closed_without_context() -> None:
    """没有上下文的调用（例如忘了包一层的离线脚本）拿到的是空集，不是全集。"""
    acl = ACLRetriever(_fake([
        _doc(r"G:\ObsidianNote\BackEndNote\a.md"),
        _doc(r"G:\ObsidianNote\LangChainNote\b.md"),
    ]))

    assert acl.invoke("anything") == []


def test_acl_retriever_always_calls_base_with_the_query() -> None:
    """过滤发生在回包阶段，query 必须原样传给底层 retriever。"""
    base = MagicMock()
    base.invoke.return_value = [_doc(r"G:\ObsidianNote\BackEndNote\a.md", "be")]
    acl = ACLRetriever(base)

    token = set_current_context(UserContext(user_id="u3", role=ROLE_USER))
    try:
        out = acl.invoke("some query")
    finally:
        reset_current_context(token)

    base.invoke.assert_called_once_with("some query")
    assert [d.page_content for d in out] == ["be"]
