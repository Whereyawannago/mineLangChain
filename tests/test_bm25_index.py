"""bm25_index 模块单测。

要点：
  - 用 MagicMock 模拟 Chroma（.get() 是普通方法，不需要 BaseRetriever 子类）
  - 用 monkeypatch 把 BM25_INDEX_PATH / BM25_HASH_PATH 重定向到 tmp_path，
    让测试隔离文件系统
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agent.rag import bm25_index
from agent.rag.bm25_index import (
    BM25_HASH_PATH,
    BM25_INDEX_PATH,
    build_bm25_retriever,
    compute_chroma_content_hash,
    load_bm25_index,
    load_or_build_bm25_retriever,
    save_bm25_index,
    tokenize_for_bm25,
)


def _mock_chroma(documents: list[str]) -> MagicMock:
    """构造一个返回固定 documents 的 Chroma mock。

    vectorstore.get(include=["documents" | "documents,metadatas" | []]) 调用方式有 3 种，
    这里统一返回一个 dict，调用方按需取 'documents' / 'metadatas' / 'ids'。
    """
    vs = MagicMock()
    metas = [{"source": f"note{i}.md"} for i in range(len(documents))]
    ids = [f"id_{i}" for i in range(len(documents))]

    def _get(include=None, **kwargs):
        # 模拟 Chroma.get 语义：ids 恒返回，include 决定额外返回哪些字段
        result = {"ids": list(ids)}
        if include:
            if "documents" in include:
                result["documents"] = list(documents)
            if "metadatas" in include:
                result["metadatas"] = list(metas)
        return result

    vs.get.side_effect = _get
    return vs


def test_tokenize_preserves_identifiers_and_segments_chinese():
    """分词：英文标识符保留，中文用 jieba。"""
    tokens = tokenize_for_bm25("SummarizationMiddleware 的 trigger 字段默认值")
    # 标识符原样
    assert "summarizationmiddleware" in tokens
    assert "trigger" in tokens
    # 中文段被切
    assert "字段" in tokens or "默认值" in tokens


def test_compute_chroma_content_hash_stable_across_calls():
    """同一内容两次调用 → 相同 hash；顺序打乱后 hash 仍相同（验证 sort）。"""
    docs = ["foo bar", "alpha beta", "gamma delta"]
    vs1 = _mock_chroma(docs)
    vs2 = _mock_chroma(list(reversed(docs)))   # 顺序反过来
    vs3 = _mock_chroma(list(docs))              # 原序再来一次

    h1 = compute_chroma_content_hash(vs1)
    h2 = compute_chroma_content_hash(vs2)
    h3 = compute_chroma_content_hash(vs3)

    assert h1 == h2 == h3
    assert len(h1) == 40  # SHA1 hex


def test_pickle_round_trip(tmp_path, monkeypatch):
    """保存后重新加载，invoke 结果与原 retriever 一致；sidecar .sha1 文件存在。"""
    idx = tmp_path / "bm25_index.pkl"
    sha = tmp_path / "bm25_index.sha1"
    monkeypatch.setattr(bm25_index, "BM25_INDEX_PATH", idx)
    monkeypatch.setattr(bm25_index, "BM25_HASH_PATH", sha)

    docs = [
        Document(page_content="python programming language", metadata={"source": "a.md"}),
        Document(page_content="java object oriented", metadata={"source": "b.md"}),
        Document(page_content="rust memory safety", metadata={"source": "c.md"}),
    ]
    retriever = build_bm25_retriever(docs, k=2)

    expected_hash = compute_chroma_content_hash(_mock_chroma([d.page_content for d in docs]))
    save_bm25_index(retriever, expected_hash)

    assert idx.exists()
    assert sha.exists()
    assert sha.read_text(encoding="utf-8") == expected_hash

    loaded = load_bm25_index()
    assert loaded is not None
    reloaded_retriever, reloaded_hash = loaded
    assert reloaded_hash == expected_hash

    # 同一 query 召回的文档内容应一致（顺序可能不同但集合一致）
    out1 = retriever.invoke("python")
    out2 = reloaded_retriever.invoke("python")
    assert {d.page_content for d in out1} == {d.page_content for d in out2}


def test_load_bm25_index_returns_none_when_missing(tmp_path, monkeypatch):
    """两文件都缺失 → 返回 None。"""
    monkeypatch.setattr(bm25_index, "BM25_INDEX_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(bm25_index, "BM25_HASH_PATH", tmp_path / "missing.sha1")
    assert load_bm25_index() is None


def test_load_or_build_rebuilds_on_hash_mismatch(tmp_path, monkeypatch):
    """hash 不一致时强制重建，并把新 hash 写到磁盘。"""
    idx = tmp_path / "bm25_index.pkl"
    sha = tmp_path / "bm25_index.sha1"
    monkeypatch.setattr(bm25_index, "BM25_INDEX_PATH", idx)
    monkeypatch.setattr(bm25_index, "BM25_HASH_PATH", sha)

    # 先用一个 mock 内容建索引
    initial_docs = ["alpha beta gamma", "delta epsilon zeta"]
    initial_vs = _mock_chroma(initial_docs)
    bm25_index.load_or_build_bm25_retriever(initial_vs, k=2)
    initial_hash = sha.read_text(encoding="utf-8")
    assert idx.exists()

    # 现在 Chroma 内容变了（新的 mock）→ 下次调用应重建
    new_docs = ["completely", "different", "content"]
    new_vs = _mock_chroma(new_docs)
    bm25_index.load_or_build_bm25_retriever(new_vs, k=2)

    new_hash = sha.read_text(encoding="utf-8")
    assert new_hash != initial_hash


def test_load_or_build_rebuilds_when_missing(tmp_path, monkeypatch):
    """pickle 不存在时应直接走重建。"""
    idx = tmp_path / "bm25_index.pkl"
    sha = tmp_path / "bm25_index.sha1"
    monkeypatch.setattr(bm25_index, "BM25_INDEX_PATH", idx)
    monkeypatch.setattr(bm25_index, "BM25_HASH_PATH", sha)
    assert not idx.exists()

    docs = ["x" * 50, "y" * 50]
    vs = _mock_chroma(docs)
    bm25_index.load_or_build_bm25_retriever(vs, k=1)

    assert idx.exists()
    assert sha.exists()


def test_load_or_build_skips_rebuild_when_hash_matches(tmp_path, monkeypatch):
    """hash 一致时不重建（pickle mtime 不更新）。"""
    idx = tmp_path / "bm25_index.pkl"
    sha = tmp_path / "bm25_index.sha1"
    monkeypatch.setattr(bm25_index, "BM25_INDEX_PATH", idx)
    monkeypatch.setattr(bm25_index, "BM25_HASH_PATH", sha)

    docs = ["same content here", "another same content"]
    vs = _mock_chroma(docs)
    bm25_index.load_or_build_bm25_retriever(vs, k=2)
    mtime_after_first = idx.stat().st_mtime

    # 同样的内容再调一次
    bm25_index.load_or_build_bm25_retriever(vs, k=2)
    mtime_after_second = idx.stat().st_mtime

    assert mtime_after_first == mtime_after_second