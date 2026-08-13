import os

from langchain_anthropic import ChatAnthropic


def build_llm() -> ChatAnthropic:
    """构造一个指向 Anthropic 兼容端点（默认 minimax 代理）的 ChatAnthropic 实例。

    配置来源（按优先级）：
        - ANTHROPIC_API_KEY  必填
        - ANTHROPIC_BASE_URL 可选；不填则走官方 Anthropic API
        - ANTHROPIC_MODEL    可选；默认 claude-sonnet-4-5
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 未设置，请检查项目根目录的 .env 文件")

    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )