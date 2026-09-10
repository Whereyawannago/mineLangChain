"""结构化输出（agent.structured）的单测。

要点：
  - 不依赖真实 LLM，只验证 Pydantic schema 自身 + ToolStrategy 工厂 + Pydantic 校验。
  - build_structured_agent() 工厂通过 patch create_agent 来验证传参是否正确。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent.structured import (
    ChatReply,
    RAGAnswer,
    WeatherReport,
    get_default_schema,
    get_schemas,
    make_response_format,
    verify_rag_sources,
)
from langchain_core.messages import ToolMessage


# ─────────────────────────── Schema 字段校验 ───────────────────────────


def test_chat_reply_minimum_valid() -> None:
    """ChatReply 只必填 content，confidence 有默认 1.0。"""
    r = ChatReply(content="你好")
    assert r.content == "你好"
    assert r.confidence == 1.0


def test_chat_reply_confidence_range() -> None:
    """confidence 必须在 [0, 1]，超出会被 Pydantic 拒掉。"""
    with pytest.raises(ValidationError):
        ChatReply(content="x", confidence=1.5)
    with pytest.raises(ValidationError):
        ChatReply(content="x", confidence=-0.1)


def test_weather_report_required_fields() -> None:
    """city / temperature / summary 必填。"""
    w = WeatherReport(city="北京", temperature=15.0, summary="晴")
    assert w.city == "北京"
    assert w.temperature == 15.0
    assert w.source == "内置知识"  # 有默认值


def test_weather_report_missing_required() -> None:
    """缺 city / temperature / summary 任一 → ValidationError。"""
    with pytest.raises(ValidationError):
        WeatherReport(city="北京")  # 缺 temperature 和 summary


def test_rag_answer_sources_default_empty() -> None:
    """不传 sources → 默认空列表，不报错。"""
    r = RAGAnswer(answer="LangChain 是 LLM 编排框架。")
    assert r.sources == []
    assert r.confidence == 1.0


def test_rag_answer_confidence_below_threshold_signals_uncertainty() -> None:
    """confidence < 0.5 表示模型自评不靠谱，下游可以二次校验。"""
    r = RAGAnswer(
        answer="不确定",
        sources=["obsidian/notes/x.md"],
        confidence=0.3,
    )
    assert r.confidence < 0.5


def test_rag_answer_sources_can_be_empty() -> None:
    """sources 是 list[str]，空列表合法。"""
    r = RAGAnswer(answer="内置知识直答", sources=[])
    assert r.sources == []


# ─────────────────────────── ToolStrategy 工厂 ───────────────────────────


def test_make_response_format_returns_tool_strategy() -> None:
    """工厂应该返回 ToolStrategy 实例。"""
    from langchain.agents.structured_output import ToolStrategy

    strategy = make_response_format(WeatherReport)
    assert isinstance(strategy, ToolStrategy)


def test_make_response_format_preserves_schema() -> None:
    """不同 schema 应得到不同的 strategy（schema 引用要传过去）。"""
    s1 = make_response_format(ChatReply)
    s2 = make_response_format(WeatherReport)
    assert s1 is not s2


def test_get_schemas_returns_all_builtins() -> None:
    """get_schemas() 应暴露本模块所有内置 schema。"""
    schemas = get_schemas()
    assert "ChatReply" in schemas
    assert "WeatherReport" in schemas
    assert "RAGAnswer" in schemas
    assert schemas["WeatherReport"] is WeatherReport


def test_get_default_schema_is_chat_reply() -> None:
    """默认 schema 必须是 ChatReply（最小可用形态）。"""
    assert get_default_schema() is ChatReply


# ─────────────────────────── build_structured_agent 工厂 ───────────────────────────


def test_build_structured_agent_calls_create_agent_with_response_format() -> None:
    """build_structured_agent 必须把 ToolStrategy 传给 create_agent 的 response_format 参数。"""
    from langchain.agents.structured_output import ToolStrategy

    with patch("agent.builder.create_agent") as mock_create:
        from agent.builder import build_structured_agent

        build_structured_agent(WeatherReport)

        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert "response_format" in kwargs
        assert isinstance(kwargs["response_format"], ToolStrategy)


def test_build_structured_agent_default_schema_when_none() -> None:
    """schema=None 时默认用 ChatReply。"""
    with patch("agent.builder.create_agent") as mock_create:
        from agent.builder import build_structured_agent

        build_structured_agent()

        kwargs = mock_create.call_args.kwargs
        # ToolStrategy 内部保存 schema 引用，做个一致性检查
        assert kwargs["response_format"] is not None


def test_build_structured_agent_omits_rag_tool_when_disabled() -> None:
    """include_rag=False 时 search_docs 工具不应出现在 tools 列表里。"""
    with patch("agent.builder.create_agent") as mock_create:
        from agent.builder import build_structured_agent

        build_structured_agent(ChatReply, include_rag=False)

        tools = mock_create.call_args.kwargs["tools"]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "search_docs" not in tool_names
        # 但 preference_tools 应该还在
        assert "save_user_preference" in tool_names
        assert "get_user_preference" in tool_names


def test_build_structured_agent_includes_rag_tool_by_default() -> None:
    """默认 include_rag=True 时 search_docs 应该被挂上。"""
    with patch("agent.builder.create_agent") as mock_create, \
         patch("agent.builder.build_embeddings"), \
         patch("agent.builder.load_vectorstore"), \
         patch("agent.builder.load_or_build_bm25_retriever"), \
         patch("agent.builder.HybridRetriever"):
        from agent.builder import build_structured_agent

        build_structured_agent(RAGAnswer)

        tools = mock_create.call_args.kwargs["tools"]
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert "search_docs" in tool_names
        assert "save_user_preference" in tool_names


# ─────────────────────────── verify_rag_sources 引用校验 ───────────────────────────


def _rag_result(messages, sources):
    """构造一个 RAGAnswer 的 invoke 结果（带来源的模型回答）。"""
    return {
        "messages": messages,
        "structured_response": RAGAnswer(answer="基于检索的回答", sources=sources),
    }


def _search_docs_tool_message(paths: list[str]) -> ToolMessage:
    """构造一条 search_docs 工具的返回消息，内容格式与 search_tool.py 一致。"""
    chunks = "\n\n---\n\n".join(
        f"[{i}] 来源: {p}\n这是第 {i} 段检索正文。" for i, p in enumerate(paths, 1)
    )
    content = (
        f"以下是从本地知识库检索到的 {len(paths)} 个相关片段。"
        f"请把它们当作参考资料回答用户问题：\n\n{chunks}"
    )
    return ToolMessage(content=content, tool_call_id="call_1", name="search_docs")


def test_verify_rag_sources_passes_when_all_citations_retrieved() -> None:
    """sources 全部在真实检索集合里 → 无伪造，返回原实例。"""
    path = "G:/notes/langchain.md"
    result = _rag_result(
        messages=[_search_docs_tool_message([path])],
        sources=[path],
    )
    verified, invalid = verify_rag_sources(result)
    assert invalid == []
    assert verified is result["structured_response"]


def test_verify_rag_sources_drops_fabricated_citation() -> None:
    """模型编造了检索里不存在的来源 → 剔除，且不污染原对象。"""
    real, fake = "G:/notes/langchain.md", "G:/notes/不存在的笔记.md"
    result = _rag_result(
        messages=[_search_docs_tool_message([real])],
        sources=[real, fake],
    )
    verified, invalid = verify_rag_sources(result)
    assert invalid == [fake]
    assert verified.sources == [real]
    # 原实例未被改动（返回的是副本）
    assert result["structured_response"].sources == [real, fake]


def test_verify_rag_sources_normalizes_path_separators() -> None:
    """模型写的反斜杠路径应与工具输出的正斜杠来源对齐。"""
    result = _rag_result(
        messages=[_search_docs_tool_message(["G:/notes/a.md"])],
        sources=["G:\\notes\\a.md"],  # 模型把分隔符写反了
    )
    verified, invalid = verify_rag_sources(result)
    assert invalid == []
    assert verified is result["structured_response"]


def test_verify_rag_sources_flags_all_when_no_retrieval_happened() -> None:
    """模型没调用检索却声明了引用 → 全部判为伪造。"""
    result = _rag_result(messages=[], sources=["G:/notes/fake.md"])
    verified, invalid = verify_rag_sources(result)
    assert invalid == ["G:/notes/fake.md"]
    assert verified.sources == []


def test_verify_rag_sources_ignores_body_lines_with_source_word() -> None:
    """检索正文里恰好出现「来源:」的文字不能被误当来源。"""
    retrieved = "G:/notes/a.md"
    chunks = f"[1] 来源: {retrieved}\n正文里说：来源: 123 这是内容不是来源。"
    content = (
        "以下是从本地知识库检索到的 1 个相关片段。"
        "请把它们当作参考资料回答用户问题：\n\n" + chunks
    )
    result = _rag_result(
        messages=[ToolMessage(content=content, tool_call_id="c1", name="search_docs")],
        sources=[retrieved, "123"],
    )
    verified, invalid = verify_rag_sources(result)
    # "123" 不是行首 [n] 来源: 前缀，不会进入基线 → 被判为伪造
    assert invalid == ["123"]
    assert verified.sources == [retrieved]


def test_verify_rag_sources_noop_for_schema_without_sources() -> None:
    """schema 不含 sources（如 ChatReply）时是 no-op，不报错。"""
    result = {
        "messages": [_search_docs_tool_message(["G:/notes/a.md"])],
        "structured_response": ChatReply(content="你好"),
    }
    verified, invalid = verify_rag_sources(result)
    assert invalid == []
    assert verified is result["structured_response"]
