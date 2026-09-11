"""agent.rag —— RAG 子包（向量检索增强生成）

按职责拆文件：

- embeddings:        本地 HuggingFace embedding 模型
- vectorstore:       Chroma 持久化（创建 / 加载）
- ingestion:         load + split + embed + store 一站式管道
- bm25_index:        稀疏索引 + 持久化 + 与 Chroma 的漂移校验
- hybrid_retriever:  dense + sparse 的 RRF 融合
- acl:               按角色过滤检索结果（fail-closed）
- search_tool:       把 retriever 包装成 @tool，让 agent 自主决定何时检索

对外暴露的工厂函数：

    from agent.rag import (
        build_embeddings,
        load_vectorstore,
        make_search_docs_tool,
        ingest_documents,
        PERSIST_DIR,
        COLLECTION_NAME,
    )
"""

from .embeddings import EMBED_MODEL_NAME, build_embeddings
from .ingestion import ingest_documents
from .search_tool import (
    ARTIFACT_SCHEMA_VERSION,
    SEARCH_DOCS_TOOL_NAME,
    build_artifact,
    format_retrieved_documents,
    make_search_docs_tool,
)
from .vectorstore import COLLECTION_NAME, PERSIST_DIR, load_vectorstore

from .acl import ACLRetriever, doc_visible_to_role, visible_docs
from .bm25_index import BM25_HASH_PATH, BM25_INDEX_PATH, load_or_build_bm25_retriever
from .hybrid_retriever import (
    DEFAULT_RRF_C,
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTS,
    DEDUP_CONTENT_PREFIX,
    HybridRetriever,
)

__all__ = [
    "EMBED_MODEL_NAME",
    "PERSIST_DIR",
    "COLLECTION_NAME",
    "build_embeddings",
    "ingest_documents",
    "load_vectorstore",
    "make_search_docs_tool",
    # search_docs 工具的输出约定（structured.py 的引用校验依赖 artifact）
    "ARTIFACT_SCHEMA_VERSION",
    "SEARCH_DOCS_TOOL_NAME",
    "build_artifact",
    "format_retrieved_documents",
    # 混合检索（向量 + BM25 + RRF）
    "BM25_HASH_PATH",
    "BM25_INDEX_PATH",
    "DEFAULT_RRF_C",
    "DEFAULT_TOP_K",
    "DEFAULT_WEIGHTS",
    "DEDUP_CONTENT_PREFIX",
    "HybridRetriever",
    "load_or_build_bm25_retriever",
    # ACL
    "ACLRetriever",
    "doc_visible_to_role",
    "visible_docs",
]