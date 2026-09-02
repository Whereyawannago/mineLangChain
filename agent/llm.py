import os

from langchain_anthropic import ChatAnthropic


def build_llm(model: str | None = None) -> ChatAnthropic:
    """构造一个指向 Anthropic 兼容端点（默认 minimax 代理）的 ChatAnthropic 实例。

    配置来源（按优先级）：
        - ANTHROPIC_API_KEY       必填
        - ANTHROPIC_BASE_URL      可选；不填则走官方 Anthropic API
        - ANTHROPIC_MODEL         可选；默认 claude-sonnet-4-5
        - ANTHROPIC_SUMMARY_MODEL 可选；默认 claude-haiku-4-5（见 build_summary_llm）

    Args:
        model: 覆盖默认模型名（优先级最高）。未传时读 `ANTHROPIC_MODEL` 环境变量。
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 未设置，请检查项目根目录的 .env 文件")

    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def build_summary_llm() -> ChatAnthropic:
    """构造用于摘要的 LLM，默认是更便宜/更快的 Haiku 4.5。

    用法：SummarizationMiddleware 触发时会调这个模型压缩老消息。
    摘要任务对推理深度要求低、对延迟和价格敏感——和主对话模型解耦后可以独立切换。

    环境变量：
        - ANTHROPIC_SUMMARY_MODEL  可选；默认 `claude-haiku-4-5-20251001`。
                                    设置成空串则回退到主模型 `ANTHROPIC_MODEL`。
    """
    override = os.getenv("ANTHROPIC_SUMMARY_MODEL")
    if override is not None and override.strip() == "":
        # 显式清空 → 回退到主模型（方便 debug）
        return build_llm()

    default_summary_model = "claude-haiku-4-5-20251001"
    return build_llm(model=override or default_summary_model)