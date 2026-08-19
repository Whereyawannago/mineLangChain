"""agent.middleware —— 中间件集合

每个中间件一个文件，便于独立维护和替换：

- model_call_limit:  模型调用上限，防止死循环
- human_in_the_loop: 人在回路，敏感工具调用前暂停
- summarization:     消息摘要，超 token 阈值时压缩老消息
"""

from .human_in_the_loop import make_human_in_the_loop_middleware
from .model_call_limit import make_model_call_limit_middleware
from .summarization import (
    MAX_CONTEXT_TOKENS,
    SUMMARY_KEEP_TOKENS,
    make_summarization_middleware,
)

__all__ = [
    "MAX_CONTEXT_TOKENS",
    "SUMMARY_KEEP_TOKENS",
    "make_human_in_the_loop_middleware",
    "make_model_call_limit_middleware",
    "make_summarization_middleware",
]