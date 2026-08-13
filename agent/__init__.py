"""agent 包 —— LangChain Agent 构造与配置。

按官方文档（docs.langchain.com）实现三类上下文管理：
    1. 短期记忆  —— checkpointer + thread_id 多轮对话
    2. 长期记忆  —— Store + 偏好工具，跨会话持久化
    3. 消息裁剪  —— @before_model 中间件按 token 阈值截断

对外只需：
    from agent import build_agent
"""

from dotenv import load_dotenv

# 模块被 import 时加载 .env，确保所有子模块能读到环境变量
load_dotenv()

from .builder import build_agent
from .context import MAX_CONTEXT_TOKENS

__all__ = ["build_agent", "MAX_CONTEXT_TOKENS"]