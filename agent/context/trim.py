"""消息裁剪（Trim messages）

通过 @before_model 中间件，在每次 LLM 调用前按 token 阈值裁剪历史消息。

为什么用启发式 token 计数（_approx_token_count），不直接调 Anthropic count_tokens API？
    - 该 API 严格校验 tool_use / tool_result 配对，
      trim 后的消息如果出现孤儿 tool_result 会被 400 拒绝。

为什么用工厂函数 make_trim_middleware，不直接写 @before_model？
    - Runtime 注入的对象没有 model 字段；目前启发式不依赖 model，
      但保留工厂模式以便未来换成基于 model 的精确计数。

官方文档：
    https://docs.langchain.com/oss/python/langchain/short-term-memory#trim-messages
"""

from langchain.agents.middleware import before_model
from langchain_core.messages import trim_messages

# 超过这个 token 数（启发式估算）就触发消息裁剪
MAX_CONTEXT_TOKENS = 2000


def _approx_token_count(messages) -> int:
    """粗略估算消息列表的总 token 数（≈ 字符数 / 3 + 元数据开销）。"""
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "")))
                    for v in block.values():
                        if not isinstance(v, (str, dict, list)):
                            total += len(str(v))
        # 每条消息加 4 个 token 的元数据开销
        total += 12
    return total // 3


def make_trim_middleware():
    """返回一个 @before_model 中间件，使用启发式 token 计数。"""
    @before_model
    def trim_history(state, runtime):
        # 没超阈值就不切
        if _approx_token_count(state["messages"]) <= MAX_CONTEXT_TOKENS:
            return None
        trimmed = trim_messages(
            state["messages"],
            max_tokens=MAX_CONTEXT_TOKENS,
            strategy="last",            # 保留最后 max_tokens 个 token
            token_counter=_approx_token_count,
            allow_partial=True,         # 允许切掉一个 tool_use/tool_result 对的中间部分
        )
        return {"messages": trimmed}

    return trim_history