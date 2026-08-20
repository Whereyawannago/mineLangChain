"""agent.rag —— RAG 子包（向量检索增强生成）

按职责拆 4 个文件：

- embeddings:   本地 HuggingFace embedding 模型
- vectorstore:  Chroma 持久化（创建 / 加载）
- ingestion:    load + split + embed + store 一站式管道
- search_tool:  把 retriever 包装成 @tool，让 agent 自主决定何时检索

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
from .search_tool import make_search_docs_tool
from .vectorstore import COLLECTION_NAME, PERSIST_DIR, load_vectorstore

__all__ = [
    "EMBED_MODEL_NAME",
    "PERSIST_DIR",
    "COLLECTION_NAME",
    "build_embeddings",
    "ingest_documents",
    "load_vectorstore",
    "make_search_docs_tool",
]