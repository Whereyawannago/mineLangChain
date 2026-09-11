"""agent 包 —— LangChain Agent 构造与配置。

按官方文档（docs.langchain.com）实现：

上下文管理（非中间件）：
    1. 短期记忆  —— checkpointer（默认 SqliteSaver，跨进程存活）+ thread_id 多轮对话
    2. 长期记忆  —— Store（默认 SqliteStore）+ 偏好工具，按 user_id 隔离、跨会话持久化
    3. 请求身份  —— UserContext / thread_id 由 user_id+session 生成（不再硬编码/随机）

中间件（agent.middleware 子包）：
    1. 模型调用上限  —— 防止 agent 死循环
    2. 人在回路      —— 敏感工具调用前暂停审批
    3. 消息摘要      —— token 超阈值时自动压缩老消息

RAG（agent.rag 子包）：
    1. 本地向量库   —— Chroma 持久化
    2. @tool 检索   —— search_docs 让 agent 自主决定何时查

⚠️ 本包 **import 时不读 .env、不配日志**（那些是入口脚本的事，见 agent/bootstrap.py）。
   入口脚本请在最开头显式调用一次 ``agent.bootstrap()``。

   唯一保留在 import 期的环境写入是 ``HF_HOME``（就在下面）：
   ``huggingface_hub`` 在 import 期就把缓存根目录固化成模块常量，而 ``.builder``
   会经 ``langchain_core → transformers`` 把它拖进来 —— 再晚就来不及了。
   这不是配置，是缓存位置，且写入前用 ``setdefault``，不覆盖宿主进程已有的值。

对外只需：
    from agent import bootstrap, build_agent

    bootstrap()          # 加载 .env / 配 HF 缓存 / 配日志
    agent = build_agent()
"""

from .bootstrap import (
    HF_CACHE_DIR,
    PROJECT_ROOT,
    bootstrap,
    configure_hf_cache,
    configure_logging,
    load_env,
)

# 必须先于下面任何 langchain 相关的 import 执行 —— 原因见模块 docstring。
configure_hf_cache()

from .builder import build_agent, build_structured_agent  # noqa: E402
from .context import UserContext, new_thread_id  # noqa: E402
from .middleware import MAX_CONTEXT_TOKENS  # noqa: E402
from .prompts import SYSTEM_PROMPT  # noqa: E402
from .structured import (  # noqa: E402
    ChatReply,
    RAGAnswer,
    WeatherReport,
    get_schemas,
    verify_rag_sources,
)

__all__ = [
    # 引导（入口脚本第一步）
    "bootstrap",
    "load_env",
    "configure_hf_cache",
    "configure_logging",
    "PROJECT_ROOT",
    "HF_CACHE_DIR",
    "build_agent",
    "build_structured_agent",
    # 提示词（单一来源，见 agent/prompts.py）
    "SYSTEM_PROMPT",
    # 身份 / 记忆持久化
    "UserContext",
    "new_thread_id",
    "MAX_CONTEXT_TOKENS",
    "ChatReply",
    "RAGAnswer",
    "WeatherReport",
    "get_schemas",
    "verify_rag_sources",
]