"""LLM 工厂 —— 统一构造 ChatAnthropic 实例。

为什么参数不能只给 model / api_key / base_url：
    早先的实现就是这三个参数，其余全走 SDK 默认值。生产上这会踩四个坑：
      - **没有 max_tokens**：模型可以一路吐到上下文上限，单次请求成本失控；
      - **没有 timeout**：上游卡住时请求线程被无限期占满，一个慢端点能拖垮整个 worker；
      - **没有 max_retries**：网络抖动 / 5xx 直接失败，用户看到裸报错；
      - **没有 temperature**：每次升级 SDK 都可能悄悄改掉默认采样行为，回答风格漂移。
    所以这四个参数全部显式给出，并允许用环境变量按部署环境调。

配置来源（按优先级：显式入参 > 环境变量 > 本文件默认值）：
    ANTHROPIC_API_KEY        必填
    ANTHROPIC_BASE_URL       可选；不填则走官方 Anthropic API
    ANTHROPIC_MODEL          可选；默认 claude-sonnet-4-5
    ANTHROPIC_SUMMARY_MODEL  可选；默认 claude-haiku-4-5（见 build_summary_llm）
    ANTHROPIC_MAX_TOKENS     单次回复上限，默认 4096
    ANTHROPIC_TEMPERATURE    采样温度，默认不传（用 SDK 默认）；摘要模型强制 0
    ANTHROPIC_TIMEOUT        请求超时秒数，默认 60
    ANTHROPIC_MAX_RETRIES    SDK 层重试次数，默认 2
"""

from __future__ import annotations

import logging
import os

from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5-20251001"

#: 单次回复 token 上限。给一个明确数字而不是留给 SDK/服务端默认，是为了让成本可预测。
DEFAULT_MAX_TOKENS = 4096
#: 摘要模型输出更短，上限压低一点，进一步省钱
DEFAULT_SUMMARY_MAX_TOKENS = 1024
#: 请求超时（秒）。没有它，一个挂住的端点会长期占用调用线程。
DEFAULT_TIMEOUT = 60.0
#: SDK 层重试次数（覆盖 429 / 5xx / 连接错误）。
DEFAULT_MAX_RETRIES = 2


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("[llm] %s=%r 不是合法数字，回落到默认 %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("[llm] %s=%r 不是合法整数，回落到默认 %s", name, raw, default)
        return default


def build_llm(
    model: str | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> ChatAnthropic:
    """构造一个指向 Anthropic 兼容端点（默认 minimax 代理）的 ChatAnthropic 实例。

    Args:
        model:       覆盖默认模型名（优先级最高）。未传时读 ``ANTHROPIC_MODEL``。
        max_tokens:  单次回复上限；None 时读 ``ANTHROPIC_MAX_TOKENS``，再缺省 4096。
        temperature: 采样温度；None 时读 ``ANTHROPIC_TEMPERATURE``，仍未设则不传给 SDK
                     （保持 SDK 默认，不强行改变既有回答风格）。
        timeout:     请求超时秒数；None 时读 ``ANTHROPIC_TIMEOUT``，再缺省 60。
        max_retries: SDK 重试次数；None 时读 ``ANTHROPIC_MAX_RETRIES``，再缺省 2。

    Raises:
        RuntimeError: ``ANTHROPIC_API_KEY`` 未设置。
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    model_name = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 未设置，请检查项目根目录的 .env 文件")

    resolved_max_tokens = (
        max_tokens if max_tokens is not None else _env_int("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    )
    resolved_timeout = (
        timeout if timeout is not None else _env_float("ANTHROPIC_TIMEOUT", DEFAULT_TIMEOUT)
    )
    resolved_retries = (
        max_retries
        if max_retries is not None
        else _env_int("ANTHROPIC_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    )

    kwargs: dict = {
        "model": model_name,
        "api_key": api_key,
        "max_tokens": resolved_max_tokens,
        "timeout": resolved_timeout,
        "max_retries": resolved_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url

    # temperature 只在明确要求时才传：SDK 默认值随版本变化，但「不传」至少保持了
    # 与升级前一致的行为；要固定风格就显式设 ANTHROPIC_TEMPERATURE。
    resolved_temperature = temperature
    if resolved_temperature is None:
        raw = (os.getenv("ANTHROPIC_TEMPERATURE") or "").strip()
        if raw:
            try:
                resolved_temperature = float(raw)
            except ValueError:
                logger.warning(
                    "[llm] ANTHROPIC_TEMPERATURE=%r 不是合法数字，忽略", raw
                )
    if resolved_temperature is not None:
        kwargs["temperature"] = resolved_temperature

    logger.info(
        "[llm] model=%s max_tokens=%s timeout=%ss max_retries=%s temperature=%s",
        model_name, resolved_max_tokens, resolved_timeout, resolved_retries,
        kwargs.get("temperature", "<sdk-default>"),
    )
    return ChatAnthropic(**kwargs)


def build_summary_llm() -> ChatAnthropic:
    """构造用于摘要的 LLM，默认是更便宜/更快的 Haiku 4.5。

    用法：SummarizationMiddleware 触发时会调这个模型压缩老消息。
    摘要任务对推理深度要求低、对延迟和价格敏感——和主对话模型解耦后可以独立切换。

    与主对话模型的三点差异（都是刻意的）：
      - 默认模型换成 Haiku（便宜、快）；
      - ``temperature=0``：摘要是确定性任务，不需要创造性，也便于回归对比；
      - ``max_tokens`` 压到 1024：摘要本身就该短。

    环境变量：
        - ANTHROPIC_SUMMARY_MODEL  可选；默认 ``claude-haiku-4-5-20251001``。
                                   设置成空串则回退到主模型 ``ANTHROPIC_MODEL``。
    """
    override = os.getenv("ANTHROPIC_SUMMARY_MODEL")
    if override is not None and override.strip() == "":
        # 显式清空 → 回退到主模型（方便 debug）
        return build_llm()

    return build_llm(
        model=override or DEFAULT_SUMMARY_MODEL,
        max_tokens=_env_int("ANTHROPIC_SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS),
        temperature=0.0,
    )
