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
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from agent.rag.search_tool import (
    SEARCH_DOCS_TOOL_NAME,
    build_artifact,
    format_retrieved_documents,
)


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
    """构造一条 search_docs 工具的返回消息，content 与 artifact 都按真实契约填。

    直接用 ``search_tool.py`` 的 ``format_retrieved_documents`` / ``build_artifact`` 生成，
    而不是在测试里手摹一份字符串 —— 手摹的那份一旦与生产格式漂移，测试会绿着通过、
    生产却已经坏了。
    """
    docs = [
        Document(page_content=f"这是第 {i} 段检索正文。", metadata={"source": p})
        for i, p in enumerate(paths, 1)
    ]
    return ToolMessage(
        content=format_retrieved_documents(docs),
        tool_call_id="call_1",
        name=SEARCH_DOCS_TOOL_NAME,
        artifact=build_artifact(docs),
    )


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


def test_verify_rag_sources_ignores_source_like_text_in_body() -> None:
    """正文里恰好出现「来源:」字样的文字不能被误当来源 —— 基线只来自 artifact。"""
    retrieved = "G:/notes/a.md"
    docs = [Document(page_content="正文里说：来源: 123 这是内容不是来源。",
                     metadata={"source": retrieved})]
    result = _rag_result(
        messages=[ToolMessage(
            content=format_retrieved_documents(docs),
            tool_call_id="c1",
            name=SEARCH_DOCS_TOOL_NAME,
            artifact=build_artifact(docs),
        )],
        sources=[retrieved, "123"],
    )
    verified, invalid = verify_rag_sources(result)
    # "123" 只存在于正文文字里，不在 artifact["sources"] → 判为伪造
    assert invalid == ["123"]
    assert verified.sources == [retrieved]


def test_verify_rag_sources_does_not_parse_content_text() -> None:
    """content 里写着来源、但 artifact 没有 → 仍然判伪造。

    这条钉住了新旧实现的分界：旧实现用正则解析 content，会把正文里的路径当基线；
    现在只认 artifact，模型就不能靠在正文里多印一行「来源: xxx」来自我背书。
    """
    claimed = "G:/notes/only-in-text.md"
    result = _rag_result(
        messages=[ToolMessage(
            content=f"来源: {claimed}\n正文",
            tool_call_id="c1",
            name=SEARCH_DOCS_TOOL_NAME,
            artifact={"schema_version": 1, "count": 0, "sources": [], "documents": []},
        )],
        sources=[claimed],
    )
    verified, invalid = verify_rag_sources(result)
    assert invalid == [claimed]
    assert verified.sources == []


def test_verify_rag_sources_treats_legacy_message_without_artifact_as_no_baseline() -> None:
    """旧 checkpoint 里恢复出来的 ToolMessage 没有 artifact → 基线为空，fail-closed。"""
    result = _rag_result(
        messages=[ToolMessage(content="[1] 来源: G:/notes/a.md", tool_call_id="c1",
                              name=SEARCH_DOCS_TOOL_NAME)],
        sources=["G:/notes/a.md"],
    )
    verified, invalid = verify_rag_sources(result)
    assert invalid == ["G:/notes/a.md"]
    assert verified.sources == []


def test_verify_rag_sources_rejects_malformed_artifact_sources() -> None:
    """artifact["sources"] 类型不对（字符串/缺失）时不能当基线，也不能抛异常。"""
    for bad_artifact in ({"sources": "G:/notes/a.md"}, {"count": 1}, None):
        result = _rag_result(
            messages=[ToolMessage(content="x", tool_call_id="c1", name=SEARCH_DOCS_TOOL_NAME,
                                  artifact=bad_artifact)],
            sources=["G:/notes/a.md"],
        )
        verified, invalid = verify_rag_sources(result)
        assert invalid == ["G:/notes/a.md"]
        assert verified.sources == []


def test_verify_rag_sources_collects_from_every_tool_message() -> None:
    """多轮检索（或另一个检索类工具）的来源都要进基线；非工具消息忽略。"""
    first, second = "G:/notes/a.md", "G:/notes/b.md"
    other_tool = ToolMessage(
        content="别的工具的输出",
        tool_call_id="c2",
        name="slow_lookup",
        artifact={"sources": ["G:/notes/c.md"]},
    )
    result = _rag_result(
        messages=[
            AIMessage(content="我先查一下"),
            _search_docs_tool_message([first]),
            _search_docs_tool_message([second]),
            other_tool,
        ],
        sources=[first, second, "G:/notes/c.md", "G:/notes/fake.md"],
    )
    verified, invalid = verify_rag_sources(result)
    assert invalid == ["G:/notes/fake.md"]
    assert verified.sources == [first, second, "G:/notes/c.md"]


def test_verify_rag_sources_noop_when_sources_empty() -> None:
    """模型没声明任何引用 → 无需校验，不报伪造。"""
    result = _rag_result(messages=[], sources=[])
    verified, invalid = verify_rag_sources(result)
    assert invalid == []
    assert verified is result["structured_response"]


def test_verify_rag_sources_noop_when_structured_response_missing() -> None:
    """没有 structured_response（模型直接回了文本）时不报错。"""
    verified, invalid = verify_rag_sources({"messages": []})
    assert verified is None
    assert invalid == []


def test_verify_rag_sources_noop_for_schema_without_sources() -> None:
    """schema 不含 sources（如 ChatReply）时是 no-op，不报错。"""
    result = {
        "messages": [_search_docs_tool_message(["G:/notes/a.md"])],
        "structured_response": ChatReply(content="你好"),
    }
    verified, invalid = verify_rag_sources(result)
    assert invalid == []
    assert verified is result["structured_response"]
