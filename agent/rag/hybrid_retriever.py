"""混合检索器：Reciprocal Rank Fusion over 多路 retriever。

为什么自己实现而不是用 langchain_classic.EnsembleRetriever：
  1. 去重键：EnsembleRetriever 默认按整 page_content 去重，但
     RecursiveCharacterTextSplitter(chunk_overlap=80) 会产生大量"差几字符"的近重复 chunk，
     整字符串对比根本识别不出来。我们用 ``source + 前200字符`` 当去重键，能稳稳命中。
  2. 单一可测单元：把权重/c/top_k/去重逻辑收在一个 ~30 行的模块里，方便单测。
  3. 不引入 langchain-classic 作为直接依赖（见 bm25_index.py 顶部注释）。

用法：
    from agent.rag.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25_retriever],
    )
    docs = retriever.invoke("query")
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# 模块级默认值：调参集中在这一处
DEFAULT_WEIGHTS: tuple[float, ...] = (0.6, 0.4)   # 向量, BM25
DEFAULT_RRF_C: int = 60                            # RRF 文献默认
DEFAULT_TOP_K: int = 4                             # 与既有 search_docs 工具 top-4 对齐
DEDUP_CONTENT_PREFIX: int = 200                    # 去重键用前 200 字符


class HybridRetriever(BaseRetriever):
    """Reciprocal Rank Fusion over a list of retrievers。

    Score(doc) = Σ w_i / (rank_i(doc) + c)
    其中 rank_i(doc) 是 doc 在第 i 个 retriever 输出中的位次（从 1 开始）。

    Args:
        retrievers: 子 retriever 列表，按顺序对应 weights。
        weights: 各路权重，长度必须等于 ``len(retrievers)``。
        c: RRF 常数，控制"低排名惩罚"的力度。
        top_k: 最终返回的文档数。
    """

    retrievers: list[BaseRetriever]
    weights: list[float] = list(DEFAULT_WEIGHTS)  # type: ignore[assignment]
    c: int = DEFAULT_RRF_C
    top_k: int = DEFAULT_TOP_K

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        weights: Sequence[float] | None = None,
        c: int = DEFAULT_RRF_C,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        weights_list = list(weights) if weights is not None else list(DEFAULT_WEIGHTS)
        if len(weights_list) != len(retrievers):
            raise ValueError(
                f"weights 长度 ({len(weights_list)}) 必须等于 retrievers 长度 ({len(retrievers)})"
            )
        if top_k <= 0:
            raise ValueError(f"top_k 必须 > 0，当前 {top_k}")
        if c <= 0:
            raise ValueError(f"c 必须 > 0，当前 {c}")

        super().__init__(  # type: ignore[call-super]
            retrievers=list(retrievers),
            weights=weights_list,
            c=c,
            top_k=top_k,
        )

    def _get_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Document]:
        """LangChain BaseRetriever 的实际调用入口。"""
        outputs: list[list[Document]] = [
            list(r.invoke(query)) for r in self.retrievers
        ]
        return _rrf_fuse(outputs, weights=self.weights, c=self.c, top_k=self.top_k)


def _dedup_key(doc: Document) -> str:
    """去重键：``source::前200字符``。

    source 经 NFKC 归一化避免路径在不同 OS 上差异。"""
    source = unicodedata.normalize("NFKC", doc.metadata.get("source", ""))
    return f"{source}::{doc.page_content[:DEDUP_CONTENT_PREFIX]}"


def _rrf_fuse(
    retriever_outputs: Sequence[Sequence[Document]],
    *,
    weights: Sequence[float],
    c: int,
    top_k: int,
) -> list[Document]:
    """RRF 融合：累加 ``w / (rank + c)`` 分数，按总分解排序，取 top_k。

    同一 chunk 在多路出现时分数相加；同 dedup_key 视为同一 chunk。
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    for output, w in zip(retriever_outputs, weights):
        for rank, doc in enumerate(output, start=1):
            if w == 0:
                # 权重 0 = 该路不贡献分数（但仍可作为唯一来源）
                if not docs:
                    key = _dedup_key(doc)
                    docs.setdefault(key, doc)
                continue
            key = _dedup_key(doc)
            scores[key] = scores.get(key, 0.0) + w / (rank + c)
            docs.setdefault(key, doc)  # first-seen 优先

    if not docs:
        return []

    ranked = sorted(
        docs.values(),
        key=lambda d: scores.get(_dedup_key(d), 0.0),
        reverse=True,
    )
    return ranked[:top_k]