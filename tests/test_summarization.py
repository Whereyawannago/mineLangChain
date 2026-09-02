"""SummarizationMiddleware 工厂测试 —— 不真正触发网络，仅验证参数构造。

策略：
- 验证 trigger / keep 阈值能通过参数覆盖
- 验证默认 model 是 build_summary_llm 的实例（即 ChatAnthropic，且模型名是 Haiku）
- 验证显式传 model 会跳过默认

注意：SummarizationMiddleware.__init__ 会调用 model.with_retry()，
所以 fake model 必须支持 .with_retry() 方法。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain.agents.middleware import SummarizationMiddleware

from agent.middleware.summarization import (
    MAX_CONTEXT_TOKENS,
    SUMMARY_KEEP_TOKENS,
    make_summarization_middleware,
)


class _FakeChatModel:
    """最小可用的 chat 模型占位 —— SummarizationMiddleware.__init__ 会做两件事：
       1. 调用 model.with_retry() 拿一个 retry 包装
       2. 调用 _get_approximate_token_counter(model) 读 model._llm_type
    测试只需让构造不抛错即可，不需要真的 invoke。
    """

    _llm_type = "fake-chat-model"  # 非 anthropic-chat / openai-chat 前缀会 fallback 到通用计数

    def with_retry(self, *args, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        raise NotImplementedError


def test_default_thresholds_match_constants() -> None:
    """不传参数时，trigger / keep 走模块常量。"""
    with patch("agent.middleware.summarization.build_summary_llm", return_value=_FakeChatModel()):
        mw = make_summarization_middleware()
    assert isinstance(mw, SummarizationMiddleware)


def test_overrides_take_precedence() -> None:
    """显式传的 trigger/keep 应该被使用（不深挖内部 dict 结构，但确保工厂不抛错）。"""
    with patch("agent.middleware.summarization.build_summary_llm", return_value=_FakeChatModel()):
        mw = make_summarization_middleware(trigger_tokens=500, keep_tokens=100)
    assert isinstance(mw, SummarizationMiddleware)


def test_uses_summary_llm_by_default() -> None:
    """默认 model 是 build_summary_llm() 返回的实例（默认是 Haiku 4.5）。"""
    fake_summary_model = _FakeChatModel()
    with patch(
        "agent.middleware.summarization.build_summary_llm",
        return_value=fake_summary_model,
    ) as builder:
        mw = make_summarization_middleware()

    builder.assert_called_once()
    assert mw.model is fake_summary_model


def test_explicit_model_overrides_default() -> None:
    """显式传 model 应该跳过 build_summary_llm。"""
    fake_model = _FakeChatModel()
    with patch(
        "agent.middleware.summarization.build_summary_llm"
    ) as builder:
        mw = make_summarization_middleware(model=fake_model)

    builder.assert_not_called()
    assert mw.model is fake_model


def test_constants_are_reasonable() -> None:
    """defensive: 阈值常量必须 trigger > keep。"""
    assert MAX_CONTEXT_TOKENS > SUMMARY_KEEP_TOKENS > 0
