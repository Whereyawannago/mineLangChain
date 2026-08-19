"""消息摘要中间件（Summarization）

用 LangChain 内置的 SummarizationMiddleware 自动压缩老消息。

工作原理（来自官方文档 oss/python/langchain/middleware/built-in）：
    当 token 数超过 trigger 阈值时：
      1. 用一个小模型（或同一个模型）对老消息生成结构化摘要
      2. 把老消息永久替换为摘要消息（persistent 写入 state）
      3. 保留最近 keep 阈值的原始消息不动

为什么不用 trim_messages 了？
    - trim 是丢弃式：超过阈值后直接砍掉前面的消息，不可逆，信息会丢
    - summarization 是压缩式：老消息被压缩成摘要保留在 state 里，未来轮还能看到要点

trigger / keep 条件类型（ContextSize 或 TriggerClause）：
    - tokens:    绝对 token 数（如 {"tokens": 2000}）
    - messages:  消息条数（如 {"messages": 20}）
    - fraction:  模型上下文窗口的比例（如 {"fraction": 0.85}），需要 langchain>=1.1 且模型有 profile 数据

官方文档：
    https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization
"""

from langchain.agents.middleware import SummarizationMiddleware

from ..llm import build_llm

# 超过这个 token 数就触发摘要
MAX_CONTEXT_TOKENS = 2000

# 摘要后保留最近多少 token 的原始消息不动
SUMMARY_KEEP_TOKENS = 400


def make_summarization_middleware() -> SummarizationMiddleware:
    """构造一个内置 SummarizationMiddleware 实例。

    - model:         用于生成摘要的模型。这里复用主模型；
                     生产环境建议换成更便宜的模型（如 Haiku）。
    - trigger:       当 token 数 >= MAX_CONTEXT_TOKENS 时触发摘要。
                     用绝对 tokens 而非 fraction，避免依赖模型 profile 数据
                     （minimax 代理上的 MiniMax-M3 没有标准 profile）。
    - keep:          摘要后保留最近 SUMMARY_KEEP_TOKENS 个 token 的原始消息。
    - token_counter: 默认用 count_tokens_approximately（启发式估算），
                     不依赖具体模型的 count_tokens API。
    """
    return SummarizationMiddleware(
        model=build_llm(),
        trigger=("tokens", MAX_CONTEXT_TOKENS),
        keep=("tokens", SUMMARY_KEEP_TOKENS),
    )