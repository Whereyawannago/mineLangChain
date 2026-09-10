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

对外只需：
    from agent import build_agent
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 模块被 import 时加载 .env，确保所有子模块能读到环境变量
load_dotenv()

# 重定向 HuggingFace 模型缓存到项目内（不占 C 盘）
_HF_HOME = Path(__file__).resolve().parent.parent / ".huggingface"
os.environ.setdefault("HF_HOME", str(_HF_HOME))

from .builder import build_agent, build_structured_agent
from .context import UserContext, new_thread_id
from .middleware import MAX_CONTEXT_TOKENS
from .structured import (
    ChatReply,
    RAGAnswer,
    WeatherReport,
    get_schemas,
    verify_rag_sources,
)

__all__ = [
    "build_agent",
    "build_structured_agent",
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