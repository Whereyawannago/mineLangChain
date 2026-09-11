"""把 Obsidian 笔记灌进 Chroma 向量库（一次性脚本）。

用法：
    uv run python demos/ingest_obsidian_notes.py
    uv run python demos/ingest_obsidian_notes.py --notes-dir D:/my-vault

数据源（优先级：--notes-dir > AGENT_OBSIDIAN_ROOT > identity.DEFAULT_OBSIDIAN_ROOT）：
    递归扫描所有 .md，包括 LangChainNote 与 BackEndNote

输出位置：
    <项目根>/data/chroma_db/（可用 AGENT_VECTORSTORE_DIR 覆盖；已 gitignore，不上传）

首次运行会自动从 HuggingFace 下载 ~93MB 的 bge-small-zh-v1.5 模型，
缓存到 <项目根>/.huggingface/。
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# 让 demos/ 子目录能找到上层 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import bootstrap  # noqa: E402
from agent.context import DEFAULT_OBSIDIAN_ROOT  # noqa: E402
from agent.rag import (  # noqa: E402
    BM25_HASH_PATH,
    BM25_INDEX_PATH,
    COLLECTION_NAME,
    PERSIST_DIR,
    build_embeddings,
    ingest_documents,
)
from agent.rag.bm25_index import _sig_path  # noqa: E402

logger = logging.getLogger(__name__)


def _resolve_notes_dir(cli_value: str | None) -> Path:
    """笔记目录解析：CLI > AGENT_OBSIDIAN_ROOT > 开发机兜底常量。

    早先这里把 ``G:\\ObsidianNote`` 写死在脚本里，identity.py 又写死了一遍 ——
    同一个路径存两处，换机器时很容易只改一处，结果是"灌库灌了 A、ACL 放行 B"。
    现在统一读同一个来源。
    """
    raw = cli_value or os.getenv("AGENT_OBSIDIAN_ROOT") or DEFAULT_OBSIDIAN_ROOT
    return Path(raw).expanduser()


def main():
    parser = argparse.ArgumentParser(description="Obsidian 笔记 → Chroma 向量库")
    parser.add_argument("--notes-dir", default=None,
                        help="笔记根目录（缺省读 AGENT_OBSIDIAN_ROOT）")
    args = parser.parse_args()

    # 入口引导：.env → HF 缓存 → 日志；Windows 默认 GBK，emoji 会被 codec 吃掉
    bootstrap()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    notes_dir = _resolve_notes_dir(args.notes_dir)

    print("=" * 60)
    print("Obsidian 笔记 → Chroma 向量库")
    print("=" * 60)
    print(f"  源: {notes_dir}")
    print(f"  目标: {PERSIST_DIR} (collection={COLLECTION_NAME})")
    print()

    if not notes_dir.exists():
        raise FileNotFoundError(
            f"找不到笔记目录: {notes_dir}\n"
            "用 --notes-dir 指定，或在 .env 里设 AGENT_OBSIDIAN_ROOT。"
        )

    # 强制重建：避免 Chroma 累积空 collection
    if PERSIST_DIR.exists():
        print(f"  ⚠️  删除旧向量库: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    # 同步清掉 BM25 索引、侧车 hash 与 HMAC 签名，让下次启动走 missing 路径。
    # 签名文件必须一起删 —— 只删 .pkl 不删 .hmac 会留下一个对不上号的孤儿签名。
    for path in (BM25_INDEX_PATH, BM25_HASH_PATH, _sig_path(BM25_INDEX_PATH)):
        if path.exists():
            print(f"  ⚠️  删除旧 BM25 索引文件: {path}")
            path.unlink()

    print(f"  → 加载 embedding 模型（首次运行会下载 ~93MB 到 {PERSIST_DIR.parent.parent / '.huggingface'}）...")
    t0 = time.time()
    embeddings = build_embeddings()
    print(f"    完成（耗时 {time.time() - t0:.1f}s）")

    print()
    print("  开始向量化...")
    vectorstore = ingest_documents(notes_dir, embeddings)

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