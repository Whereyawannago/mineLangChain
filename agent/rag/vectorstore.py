"""Chroma 向量库的持久化。

设计要点：
  - PERSIST_DIR 固定在项目内 data/chroma_db/（gitignore 已排除）
  - COLLECTION_NAME 固定一个名称，确保 build_agent() 每次复用同一 collection
  - build_agent() 调用 load_vectorstore()（不创建）；
    demos/ingest_obsidian_notes.py 负责首次创建

⚠️ Chroma 的 from_documents 每次会创建新的 collection UUID，所以不要在
   build_agent() 里调它，否则每次启动都会重复建库。
"""

import os
from pathlib import Path

from langchain_chroma import Chroma

# 持久化目录：项目根 / data / chroma_db
_PERSIST_DIR = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db"))
)
PERSIST_DIR = _PERSIST_DIR

# collection 名：固定一个，确保 load_vectorstore() 能复用同一 collection
COLLECTION_NAME = "mineLangChain"


def load_vectorstore(embeddings) -> Chroma:
    """加载已存在的 Chroma 向量库。

    用于 build_agent()。如果 PERSIST_DIR 不存在，会抛出异常——
    此时需要先跑 demos/ingest_obsidian_notes.py 灌库。
    """
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {PERSIST_DIR}\n"
            "请先运行：uv run python demos/ingest_obsidian_notes.py"
        )
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )