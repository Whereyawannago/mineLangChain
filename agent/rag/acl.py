"""RAG 检索 ACL —— 按角色过滤返回的文档片段。

设计要点：
  - 过滤在 **retriever 层**完成（不是 search_docs 工具层），因为如果放到工具层，
    过滤后的文档可能不够（默认 top_k=4，filter 完只剩 1 条），需要更复杂的"再多取一些直到凑够"逻辑。
  - 用户身份通过 ContextVar 传递（retriever 是 builder 期构造，没有 Thread/Request 上下文）。
  - 角色 → 允许的目录集合定义在 identity.py 里（admin 放行全集；user 仅限 BackEndNote）。
  - 判定基于 chunk 的 ``metadata["source"]`` 路径前缀：不需要在 ingest 时给每条 chunk 写 ACL tag，
    策略变了改 env 变量即可，不需要重灌库。

安全基线（**fail-closed**）：
    拿不到请求上下文时一律返回空集合，而不是原样放行。早先的实现是"没上下文就原样返回"，
    理由是"builder 期的预检调用走这里"——但只要有一个请求路径忘了 ``set_current_context``，
    整个知识库就直接泄露，而且不会有任何报错。默认拒绝 + 显式放行才是正确的姿势：
    确实需要看全集的调用方（离线评测 / 灌库脚本 / 管理后台）必须自己写明
    ``visible_docs(docs, role=ROLE_ADMIN)``，让"越权"变成一处可审计的显式代码。

性能：is_relative_to 是 O(depth) 单调用，每次 query 顶多比对 user_allowed_dirs() 的长度个
根目录 —— 几乎免费。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agent.context.identity import (
    ROLE_ADMIN,
    Role,
    allowed_paths_for_role,
    current_context,
)

logger = logging.getLogger(__name__)


def _normalize(path_str: str) -> Path:
    """metadata.source 可能是 Windows 反斜杠，统一为正斜杠再解析。"""
    return Path(path_str.replace("\\", "/")).resolve()


def doc_visible_to_role(doc: Document, *, role: Role, allowed: tuple[Path, ...]) -> bool:
    """判断一条 chunk 是否对当前 (role, allowed_paths) 可见。

    - admin：始终可见；
    - user：doc 的 source 路径必须落在 allowed 任一根目录下（用 is_relative_to 防 ../逃逸）。
    """
    if role == ROLE_ADMIN:
        return True
    if not allowed:
        return False  # 非 admin 且没有任何允许路径 → 一律拒绝（更安全的默认）
    src = doc.metadata.get("source")
    if not src:
        return False
    try:
        norm = _normalize(src)
    except (OSError, ValueError):
        return False
    return any(norm.is_relative_to(root) for root in allowed)


def visible_docs(
    docs: Iterable[Document],
    *,
    role: Role | None = None,
    allowed: tuple[Path, ...] | None = None,
) -> list[Document]:
    """按当前上下文过滤文档。**拿不到上下文时返回空列表（fail-closed）**。

    Args:
        docs:    待过滤的文档。
        role:    显式指定角色。None 时从 ContextVar 里的 UserContext 取。
        allowed: 显式指定白名单根目录；仅在 ``role`` 也显式给出时才有意义。

    需要放行全集的合法场景（离线评测、灌库、管理后台）请显式传 ``role=ROLE_ADMIN``，
    不要依赖"没上下文就放行"这种隐式行为。
    """
    materialized = list(docs)
    if role is None:
        ctx = current_context()
        if ctx is None:
            logger.warning(
                "[acl] 当前请求没有 UserContext，fail-closed 拦下 %d 条文档。"
                "调用方需 set_current_context(...) 或显式传 role=ROLE_ADMIN。",
                len(materialized),
            )
            return []
        role = ctx.role
        allowed = allowed_paths_for_role(role, ctx.extra_allowed_paths)
    if role == ROLE_ADMIN:
        return materialized
    if allowed is None:
        allowed = allowed_paths_for_role(role)
    visible = [d for d in materialized if doc_visible_to_role(d, role=role, allowed=allowed)]
    if len(visible) != len(materialized):
        logger.info(
            "[acl] role=%s 过滤 %d → %d 条", role, len(materialized), len(visible)
        )
    return visible


class ACLRetriever(BaseRetriever):
    """包装任意 BaseRetriever，在结果回包时按当前上下文的角色过滤。

    与 hybrid_retriever.py 的 HybridRetriever 同模式，但放这里（rag/acl.py）—— 关注点分离：
    HybridRetriever 关心"怎么把多路融合"，ACLRetriever 关心"按身份放行"。

    显式继承 ``BaseRetriever``（而不是只实现 invoke / _get_relevant_documents 的鸭子类型）：
    LangChain 内部多处会做 ``isinstance(x, BaseRetriever)`` 判定，鸭子类型会被判失败，
    导致某些 chain / middleware 静默走另一条分支。
    """

    #: 被包装的 retriever。类型放宽到 Any：既接受 HybridRetriever / VectorStoreRetriever，
    #: 也接受测试里的替身对象；pydantic 不做二次校验，避免嵌套模型的深拷贝开销。
    base: Any

    def __init__(self, base: Any, **kwargs: Any) -> None:
        super().__init__(base=base, **kwargs)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """BaseRetriever 的实际调用入口：先取，再按身份过滤。"""
        docs = self.base.invoke(query)
        visible = visible_docs(docs)
        logger.debug(
            "[acl] ACLRetriever query=%r 命中 %d 条，放行 %d 条",
            query[:40], len(docs), len(visible),
        )
        return visible
