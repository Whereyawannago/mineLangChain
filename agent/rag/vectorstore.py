"""Chroma 向量库的持久化。

设计要点：
  - PERSIST_DIR 默认在项目内 data/chroma_db/（gitignore 已排除），可用
    环境变量 ``AGENT_VECTORSTORE_DIR`` 覆盖（容器里挂卷 / 多环境共用一份镜像时要用）
  - COLLECTION_NAME 固定一个名称，确保 build_agent() 每次复用同一 collection
  - build_agent() 调用 load_vectorstore()（不创建）；
    demos/ingest_obsidian_notes.py 负责首次创建

⚠️ Chroma 的 from_documents 每次会创建新的 collection UUID，所以不要在
   build_agent() 里调它，否则每次启动都会重复建库。
⚠️ PERSIST_DIR 是 **import 期常量**（Chroma 一旦打开就不应再换目录），
   改环境变量需要重启进程。
"""

import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 持久化目录：默认 项目根 / data / chroma_db，可用 AGENT_VECTORSTORE_DIR 覆盖
_DEFAULT_PERSIST_DIR = _PROJECT_ROOT / "data" / "chroma_db"
PERSIST_DIR = Path(
    os.getenv("AGENT_VECTORSTORE_DIR") or _DEFAULT_PERSIST_DIR
).expanduser()

# collection 名：固定一个，确保 load_vectorstore() 能复用同一 collection
COLLECTION_NAME = os.getenv("AGENT_VECTORSTORE_COLLECTION") or "mineLangChain"


def load_vectorstore(embeddings: Embeddings) -> Chroma:
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