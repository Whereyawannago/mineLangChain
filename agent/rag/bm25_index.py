"""BM25 稀疏索引 + 持久化 + 与 Chroma 的漂移校验。

设计要点：
  - 单数据源：所有 chunk 从 Chroma `vectorstore.get(include=["documents"])["documents"]` 读出，
    避免维护两份切分逻辑。
  - 内容 hash 校验：SHA1(sorted(texts)) 持久化在 pickle 里，启动时重算比对，不一致就重建。
    这是处理 "Chroma 重建了但 pkl 没删" 的唯一可靠手段。
  - 原子写：tmp → os.replace，避免 pickle 写到一半被进程杀掉导致下次启动反序列化失败。
  - jieba 分词 + 保留英文/下划线标识符，让 BM25 召回"中文词"和"SummarizationMiddleware"都生效。

安全（pickle 反序列化等价于任意代码执行，必须当攻击面处理）：
  - **白名单 Unpickler**：`_RestrictedUnpickler.find_class` 只放行构造 BM25Retriever 真正
    需要的那几个符号，其它一律 `UnpicklingError`。这是防 RCE 的主手段 —— 攻击者就算能
    改写 .pkl（共享卷 / 备份恢复 / CI 产物被投毒），也没法让 `os.system` 之类的类被解析出来。
  - **HMAC-SHA256 侧车签名**：`<index>.hmac` 存 pickle 字节的签名，加载前比对。
    配了 `AGENT_BM25_SIGNING_KEY` 才是真「防篡改」；没配时用一个公开的派生 key，
    只能发现半截写入 / 磁盘损坏（完整性），不能防蓄意伪造（真实性）。
    **生产环境必须配这个环境变量**，否则会打 WARNING。
  - 任何一步校验失败都不抛异常，而是走「重建」路径 —— 缓存文件不可信时的正确降级
    是重新生成，不是让进程起不来。

公开 API：
  BM25_INDEX_PATH, BM25_HASH_PATH       # 持久化文件路径（.hmac 侧车由 index 路径派生）
  tokenize_for_bm25(text)               # preprocess_func for BM25Retriever
  build_bm25_retriever(docs, k=8)       # 从 Document 列表构造
  compute_chroma_content_hash(vectorstore)
  save_bm25_index(retriever, hash)      # 原子写 + 签名
  load_bm25_index()                      # (retriever, hash) | None
  load_or_build_bm25_retriever(...)      # 主入口（hash 校验 + 自动重建）
"""

from __future__ import annotations

import hashlib
import hmac
import io
import logging
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

# ────────────────────────────── pickle 反序列化加固 ──────────────────────────────

#: 允许从 pickle 里解析出来的全局符号白名单（(module, name) 二元组）。
#: 这份清单是用 ``pickletools`` 扫真实产物得到的**全集**——BM25Retriever 的 pickle 只
#: 引用这 4 个符号。宁可漏（漏了就重建，功能不受影响）也不能多（多了就是 RCE 面）。
_ALLOWED_UNPICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset({
    ("agent.rag.bm25_index", "tokenize_for_bm25"),
    ("langchain_community.retrievers.bm25", "BM25Retriever"),
    ("langchain_core.documents.base", "Document"),
    ("rank_bm25", "BM25Okapi"),
})

#: 未配置 AGENT_BM25_SIGNING_KEY 时用的公开派生 key —— 只保证完整性，不保证真实性。
_LOCAL_DEV_SIGNING_KEY = b"mineLangChain/bm25-index/unauthenticated-local-key"

# 只警告一次，避免每次加载都刷屏
_warned_unauthenticated = False


class _RestrictedUnpickler(pickle.Unpickler):
    """只放行白名单符号的 Unpickler。

    pickle 的 ``REDUCE`` / ``STACK_GLOBAL`` 会 import 任意模块并取任意属性，等价于
    任意代码执行。覆盖 ``find_class`` 是标准的最小权限做法。
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_UNPICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f"禁止反序列化 {module}.{name}（不在 BM25 索引白名单里）"
            )
        return super().find_class(module, name)


def _sig_path(index_path: Path) -> Path:
    """签名侧车路径。**调用期派生**而不是模块常量 —— 测试会 monkeypatch BM25_INDEX_PATH。"""
    return index_path.with_name(index_path.name + ".hmac")


def _signing_key() -> tuple[bytes, bool]:
    """返回 ``(key, authenticated)``。

    ``AGENT_BM25_SIGNING_KEY`` 配了 → authenticated=True，签名能防篡改；
    没配 → 用公开派生 key，只能发现文件损坏 / 半截写入，首次会打一条 WARNING。
    """
    global _warned_unauthenticated
    secret = (os.getenv("AGENT_BM25_SIGNING_KEY") or "").strip()
    if secret:
        return secret.encode("utf-8"), True
    if not _warned_unauthenticated:
        _warned_unauthenticated = True
        logger.warning(
            "[bm25_index] 未设置 AGENT_BM25_SIGNING_KEY：索引签名只保证完整性、不防篡改。"
            "生产环境请配置一个独立密钥。"
        )
    return _LOCAL_DEV_SIGNING_KEY, False


def _compute_signature(payload: bytes, index_path: Path) -> str:
    """对 pickle 字节算 HMAC-SHA256（hex）。签名里带上文件路径，防跨位置拷贝复用。"""
    key, _ = _signing_key()
    mac = hmac.new(key, digestmod=hashlib.sha256)
    mac.update(str(index_path).encode("utf-8"))
    mac.update(b"\x00")
    mac.update(payload)
    return mac.hexdigest()


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


def _atomic_write_bytes(path: Path, blob: bytes) -> None:
    """先写 .tmp 再 os.replace —— 避免写到一半被杀掉留下破损文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_bm25_index(retriever: BM25Retriever, content_hash: str) -> None:
    """原子写 pickle + HMAC 签名侧车。

    同时把 content_hash 写到侧车 .sha1 文件，方便 `Get-Content` 直接看。
    三个文件都是各自原子写；中途挂掉最多导致下次加载校验失败 → 重建，不会读到半截 pickle。
    """
    payload = {
        "_schema_version": _SCHEMA_VERSION,
        "retriever": retriever,
        "content_hash": content_hash,
    }
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(BM25_INDEX_PATH, blob)
    signature = _compute_signature(blob, BM25_INDEX_PATH)
    _atomic_write_bytes(_sig_path(BM25_INDEX_PATH), signature.encode("ascii"))
    _atomic_write_bytes(BM25_HASH_PATH, content_hash.encode("utf-8"))
    logger.debug("[bm25_index] saved %d bytes, sig=%s", len(blob), signature[:12])


def load_bm25_index() -> tuple[BM25Retriever, str] | None:
    """校验签名 → 白名单反序列化 → schema 版本校验。任一环节不过就返回 None（调用方重建）。

    Returns:
        ``(retriever, content_hash)`` 或 ``None``。
    """
    index_path = BM25_INDEX_PATH
    sig_path = _sig_path(index_path)
    if not (index_path.exists() and BM25_HASH_PATH.exists()):
        return None

    try:
        blob = index_path.read_bytes()
    except OSError as exc:
        logger.warning("[bm25_index] 读 pickle 失败 (%s)；将重建", exc)
        return None

    # ── 1. 签名校验：没签名 / 签名不对 → 不允许进入反序列化
    if not sig_path.exists():
        logger.warning("[bm25_index] 缺少签名侧车 %s；视为不可信，将重建", sig_path)
        return None
    try:
        expected_sig = sig_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("[bm25_index] 读签名失败 (%s)；将重建", exc)
        return None
    if not hmac.compare_digest(expected_sig, _compute_signature(blob, index_path)):
        logger.error(
            "[bm25_index] 签名校验失败（%s 被改动，或 AGENT_BM25_SIGNING_KEY 变了）；"
            "已阻止反序列化，将重建", index_path,
        )
        return None

    # ── 2. 白名单反序列化：防 pickle RCE
    try:
        payload = _RestrictedUnpickler(io.BytesIO(blob)).load()
    except Exception as exc:
        logger.warning("[bm25_index] failed to load pickle (%s); will rebuild", exc)
        return None

    # ── 3. schema 版本校验
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
        else (
            "hash_mismatch"
            if cached is not None
            # cached is None 有两种成因：文件本来就不在，或者在但没通过签名/白名单校验。
            # 分开报 reason，否则“索引被篡改”会被误读成“首次启动”。
            else ("missing" if not BM25_INDEX_PATH.exists() else "untrusted")
        )
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