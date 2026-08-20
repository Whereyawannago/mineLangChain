"""文档摄取管道：load + split + embed + store。

设计要点：
  - load_markdown_files: 用 langchain_community 的 DirectoryLoader
  - split_documents: RecursiveCharacterTextSplitter，中英文友好分隔符
  - ingest_documents: 把上面两步串起来，返回 Chroma 实例

注意：ingest_documents 每次调用都会创建新 collection UUID。
      demos 脚本里会用 shutil.rmtree 先清空旧库再灌，保证幂等。
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from .vectorstore import COLLECTION_NAME, PERSIST_DIR
from langchain_chroma import Chroma

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def load_markdown_files(directory: str | Path) -> list[Document]:
    """加载目录下所有 .md 文件，每个文件一个 Document。

    Args:
        directory: 包含 markdown 文件的目录路径。

    Returns:
        Document 列表，metadata["source"] 保留相对路径。
    """
    loader = DirectoryLoader(
        str(directory),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load()


def split_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """把长文档切成小块，保留 metadata。

    separators 顺序对中文友好：双换行 > 单换行 > 句号 > 问号 > 感叹号 > 空格。
    """
    # LangChain 1.x 的 RecursiveCharacterTextSplitter 在 langchain_text_splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_documents(source_dir: str | Path, embeddings) -> Chroma:
    """一站式管道：load → split → embed → store。

    ⚠️ 每次调用都会用 from_documents 创建新 collection，
       所以通常在 demos 脚本里配合 shutil.rmtree(PERSIST_DIR) 使用。

    Args:
        source_dir:  文档目录。
        embeddings:  build_embeddings() 返回的实例。

    Returns:
        Chroma 实例（已写入 PERSIST_DIR）。
    """
    print(f"  → 加载文档：{source_dir}")
    docs = load_markdown_files(source_dir)
    print(f"  → 原始文档数：{len(docs)}")

    splits = split_documents(docs)
    print(f"  → 切分后块数：{len(splits)}")

    print(f"  → 写入 Chroma：{PERSIST_DIR} (collection={COLLECTION_NAME})")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )
    return vectorstore