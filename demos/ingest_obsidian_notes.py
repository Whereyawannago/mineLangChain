"""把 Obsidian 笔记灌进 Chroma 向量库（一次性脚本）。

用法：
    uv run python demos/ingest_obsidian_notes.py

数据源（可改 OBSIDIAN_NOTES_DIR 切换）：
    G:\\ObsidianNote\\LangChainNote\\langChain\\

输出位置：
    K:\\code\\langChainExample\\data\\chroma_db\\
    （已 gitignore，项目内，不上传）

首次运行会自动从 HuggingFace 下载 ~93MB 的 bge-small-zh-v1.5 模型，
缓存到 K:\\code\\langChainExample\\.huggingface\\。
"""

import shutil
import sys
import time
from pathlib import Path

# 让 demos/ 子目录能找到上层 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.rag import (
    COLLECTION_NAME,
    PERSIST_DIR,
    build_embeddings,
    ingest_documents,
)

# Obsidian 笔记位置（G 盘）
OBSIDIAN_NOTES_DIR = Path(r"G:\ObsidianNote\LangChainNote\langChain")


def main():
    print("=" * 60)
    print("Obsidian 笔记 → Chroma 向量库")
    print("=" * 60)
    print(f"  源: {OBSIDIAN_NOTES_DIR}")
    print(f"  目标: {PERSIST_DIR} (collection={COLLECTION_NAME})")
    print()

    if not OBSIDIAN_NOTES_DIR.exists():
        raise FileNotFoundError(f"找不到笔记目录: {OBSIDIAN_NOTES_DIR}")

    # 强制重建：避免 Chroma 累积空 collection
    if PERSIST_DIR.exists():
        print(f"  ⚠️  删除旧向量库: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    print(f"  → 加载 embedding 模型（首次运行会下载 ~93MB 到 {PERSIST_DIR.parent.parent / '.huggingface'}）...")
    t0 = time.time()
    embeddings = build_embeddings()
    print(f"    完成（耗时 {time.time() - t0:.1f}s）")

    print()
    print("  开始向量化...")
    vectorstore = ingest_documents(OBSIDIAN_NOTES_DIR, embeddings)

    # Chroma 0.5+ 的 collection count 需要通过 _collection
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = "未知"
    print()
    print(f"  ✅ 完成，共写入 {count} 个文档块")
    print(f"  持久化位置：{PERSIST_DIR}")
    print()
    print("现在可以运行 main.py 测试：")
    print("  uv run python main.py")
    print("试试问：")
    print('  - "langchain-rag 这个 skill 里讲了哪些内容？"')
    print('  - "哪些中间件实现了防死循环？"')


if __name__ == "__main__":
    main()