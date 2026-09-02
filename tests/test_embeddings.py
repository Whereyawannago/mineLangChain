"""HuggingFaceEmbeddings 适配层测试 —— 验证：

1. 工厂函数返回 HuggingFaceEmbeddings 实例
2. embed_query 返回向量长度与模型匹配（bge-small-zh-v1.5 = 512）
3. embed_documents 支持多条输入，且向量长度一致

⚠️ 本测试会真的加载模型（首次约 93MB），需要机器能访问 HF Hub。
   如果环境无网，本测试会失败 —— 这是当前实现的已知前置条件。
"""

from __future__ import annotations

import pytest

from agent.rag.embeddings import EMBED_MODEL_NAME, build_embeddings


# bge-small-zh-v1.5 固定的向量维度
EXPECTED_DIM = 512


@pytest.fixture(scope="module")
def emb():
    """模块级共享一个实例 —— sentence-transformers 加载开销大。"""
    return build_embeddings()


def test_factory_returns_hf_embeddings(emb) -> None:
    """build_embeddings() 应返回 langchain_huggingface 的实例。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    assert isinstance(emb, HuggingFaceEmbeddings)


def test_embed_query_dim(emb) -> None:
    """embed_query 返回的向量维度与模型匹配。"""
    vec = emb.embed_query("LangChain 是一个 LLM 编排框架。")
    assert isinstance(vec, list)
    assert len(vec) == EXPECTED_DIM
    # bge 默认归一化，所以 L2 范数应接近 1
    norm = sum(x * x for x in vec) ** 0.5
    assert 0.9 < norm < 1.1


def test_embed_documents_consistency(emb) -> None:
    """embed_documents 应对每条文本返回相同维度的向量。"""
    texts = [
        "LangChain 提供 RAG、Agent、Middleware 等抽象。",
        "LangGraph 用图状态机建模多步 agent。",
        "向量检索依赖 Embedding 把文本变成稠密向量。",
    ]
    vecs = emb.embed_documents(texts)
    assert len(vecs) == len(texts)
    for v in vecs:
        assert len(v) == EXPECTED_DIM


def test_default_model_name() -> None:
    """默认走 bge-small-zh-v1.5，dim=512。"""
    assert EMBED_MODEL_NAME == "BAAI/bge-small-zh-v1.5"