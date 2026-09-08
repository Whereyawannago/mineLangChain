"""BM25 稀疏索引 + 持久化 + 与 Chroma 的漂移校验。

设计要点：
  - 单数据源：所有 chunk 从 Chroma `vectorstore.get(include=["documents"])["documents"]` 读出，
    避免维护两份切分逻辑。
  - 内容 hash 校验：SHA1(sorted(texts)) 持久化在 pickle 里，启动时重算比对，不一致就重建。
    这是处理 "Chroma 重建了但 pkl 没删" 的唯一可靠手段。
  - 原子写：tmp → os.replace，避免 pickle 写到一半被进程杀掉导致下次启动反序列化失败。
  - jieba 分词 + 保留英文/下划线标识符，让 BM25 召回"中文词"和"SummarizationMiddleware"都生效。

公开 API：
  BM25_INDEX_PATH, BM25_HASH_PATH       # 持久化文件路径
  tokenize_for_bm25(text)               # preprocess_func for BM25Retriever
  build_bm25_retriever(docs, k=8)       # 从 Document 列表构造
  compute_chroma_content_hash(vectorstore)
  save_bm25_index(retriever, hash)      # 原子写
  load_bm25_index()                      # (retriever, hash) | None
  load_or_build_bm25_retriever(...)      # 主入口（hash 校验 + 自动重建）
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .vectorstore import PERSIST_DIR

if TYPE_CHECKING:
    from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# 持久化文件：与 chroma_db 同级，方便一起管理
BM25_INDEX_PATH: Path = PERSIST_DIR.parent / "bm25_index.pkl"
BM25_HASH_PATH: Path = PERSIST_DIR.parent / "bm25_index.sha1"

# Pickle payload schema version —— 未来 pickle 结构变更时用来检测
_SCHEMA_VERSION = 1

# 分词正则：交替切出"标识符段"和"连续中文段"
_RE_TOKEN_SPLIT = re.compile(r"([A-Za-z_][A-Za-z0-9_]+|[一-鿿]+)")
_RE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """BM25 专用分词：NFKC + lower + jieba 切中文，标识符段原样保留。

    jieba 默认对英文标识符识别一般，所以先正则把 `BM25Retriever` 这类
    token 摘出来原样保留，再让 jieba 只切剩下的中文段。
    """
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for chunk in _RE_TOKEN_SPLIT.split(text):
        if not chunk:
            continue
        if _RE_IDENTIFIER.fullmatch(chunk):
            tokens.append(chunk)
        else:
            tokens.extend(jieba.cut(chunk, cut_all=False))
    return [t for t in tokens if t.strip()]


def build_bm25_retriever(docs: list[Document], k: int = 8) -> BM25Retriever:
    """从 Document 列表构造 BM25Retriever。

    Args:
        docs: 已经切好块的 Document 列表（与 Chroma 内容一致）。
        k: 每次查询返回的候选数。

    Returns:
        ``BM25Retriever``，可直接传 ``invoke(query)``。
    """
    if not docs:
        raise ValueError("build_bm25_retriever: docs 不能为空")
    return BM25Retriever.from_documents(
        documents=docs,
        preprocess_func=tokenize_for_bm25,
        k=k,
    )


def compute_chroma_content_hash(vectorstore: "Chroma") -> str:
    """计算 Chroma 当前内容的 SHA1。

    算法：把所有 chunk 的 page_content 排序后用 \\n 拼接 → SHA1。
    排序保证了 Chroma 内部 ID 顺序变化不会改变 hash。
    """
    raw = vectorstore.get(include=["documents"])
    documents = raw.get("documents", []) if isinstance(raw, dict) else []
    joined = "\n".join(sorted(documents))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def save_bm25_index(retriever: BM25Retriever, content_hash: str) -> None:
    """原子写：先写 .tmp 再 os.replace。

    同时把 content_hash 写到侧车 .sha1 文件，方便 `Get-Content` 直接看。
    """
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_schema_version": _SCHEMA_VERSION,
        "retriever": retriever,
        "content_hash": content_hash,
    }
    tmp_path = BM25_INDEX_PATH.with_suffix(BM25_INDEX_PATH.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, BM25_INDEX_PATH)
    BM25_HASH_PATH.write_text(content_hash, encoding="utf-8")


def load_bm25_index() -> tuple[BM25Retriever, str] | None:
    """读 pickle 与侧车 .sha1。两文件都存在才返回，否则视为缺失。

    Returns:
        ``(retriever, content_hash)`` 或 ``None``。
    """
    if not (BM25_INDEX_PATH.exists() and BM25_HASH_PATH.exists()):
        return None
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        logger.warning("[bm25_index] failed to load pickle (%s); will rebuild", exc)
        return None
    if not isinstance(payload, dict) or payload.get("_schema_version") != _SCHEMA_VERSION:
        logger.warning("[bm25_index] pickle schema mismatch; will rebuild")
        return None
    retriever: BM25Retriever = payload["retriever"]
    content_hash: str = payload["content_hash"]
    return retriever, content_hash


def load_or_build_bm25_retriever(
    vectorstore: "Chroma",
    k: int = 8,
    force_rebuild: bool = False,
) -> BM25Retriever:
    """主入口：根据 Chroma 内容 hash 决定加载或重建。

    Args:
        vectorstore: 已加载的 Chroma 实例。
        k: 候选数。
        force_rebuild: 跳过 hash 校验，强制重建（用于调试）。

    Returns:
        ``BM25Retriever``。
    """
    current_hash = compute_chroma_content_hash(vectorstore)

    cached = None if force_rebuild else load_bm25_index()

    if cached is not None and cached[1] == current_hash:
        retriever, _ = cached
        logger.info(
            "[bm25_index] loaded chunks=%d hash=%s",
            _count_chunks(vectorstore),
            current_hash[:12],
        )
        return retriever

    reason = (
        "force"
        if force_rebuild
        else ("missing" if cached is None else "hash_mismatch")
    )
    logger.info("[bm25_index] rebuild reason=%s", reason)
    return _rebuild_and_save(vectorstore, k, current_hash)


def _rebuild_and_save(
    vectorstore: "Chroma",
    k: int,
    content_hash: str,
) -> BM25Retriever:
    """从 Chroma 重建 BM25 索引并保存。"""
    raw = vectorstore.get(include=["documents", "metadatas"])
    documents_raw = raw.get("documents", []) if isinstance(raw, dict) else []
    metadatas_raw = raw.get("metadatas", []) if isinstance(raw, dict) else []
    # 对齐 documents / metadatas 长度（Chroma 应保证一致，但保险起见）
    n = min(len(documents_raw), len(metadatas_raw))
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(documents_raw[:n], metadatas_raw[:n])
    ]
    retriever = build_bm25_retriever(docs, k=k)
    save_bm25_index(retriever, content_hash)
    logger.info(
        "[bm25_index] rebuilt chunks=%d hash=%s",
        n,
        content_hash[:12],
    )
    return retriever


def _count_chunks(vectorstore: "Chroma") -> int:
    """取 Chroma 的 chunk 数（用于日志）。"""
    raw = vectorstore.get(include=[])
    ids = raw.get("ids", []) if isinstance(raw, dict) else []
    return len(ids)