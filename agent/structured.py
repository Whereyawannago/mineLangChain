"""结构化输出（Structured output）

把 LLM 的自由文本回复强制收敛到 Pydantic schema，让下游业务代码可以直接 `.字段` 取值。

官方文档：
    https://docs.langchain.com/oss/python/langchain/agents#response-format

为什么是「ToolStrategy」而不是「JSON mode」？
  - JSON mode：要求模型在 content 里直接吐 JSON 字符串，再让 SDK 解析。
    - 缺点：模型可能在 JSON 里夹带解释文字、字段缺失、类型错乱，解析脆弱。
  - ToolStrategy：把 schema 声明为「伪工具调用」，强迫模型用结构化工具调用输出。
    - 优点：复用 tool calling 机制，模型「更听话」；返回键直接在 result["structured_response"] 里。
    - 缺点：仍依赖模型支持 tool calling（Anthropic / OpenAI 都支持）。

本模块约定：
  - ChatReply       通用兜底 schema（content / confidence）
  - WeatherReport   天气查询示例（city / temperature / summary / source）
  - RAGAnswer       检索增强问答（answer / sources / confidence）
  - make_response_format(schema) 把 schema 包成 ToolStrategy 实例
  - verify_rag_sources(result)   剔除 RAGAnswer 里被模型伪造的引用来源（后校验）
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field


# ────────────────────────────── Schema 定义 ──────────────────────────────


class ChatReply(BaseModel):
    """通用对话回复：所有结构化输出都至少包含的字段。

    用作其他 schema 的基类不是必须的（多继承 + 字段命名冲突会很麻烦），
    这里只是定义一个「最小可用形态」——演示 ToolStrategy 的最小用法。
    """

    content: str = Field(description="回复正文，给用户看的最终答案")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="模型自评置信度（0=完全没把握，1=非常确定）；用于下游是否二次校验",
    )


class WeatherReport(BaseModel):
    """天气查询的结构化结果。

    演示场景：用户问「北京今天天气怎么样」→ agent 调天气工具（或直接用内置知识）
    → 必须返回这个 schema，下游业务代码直接 `report.city / temperature / summary`。
    """

    city: str = Field(description="城市名（用中文）")
    temperature: float = Field(description="当前气温（摄氏度）")
    summary: str = Field(description="一句话天气总结，例如「晴转多云，气温 15-22°C」")
    source: str = Field(
        default="内置知识",
        description="数据来源说明，例如「调用了 xxx 天气 API」或「内置知识（截止 2026 年）",
    )


class RAGAnswer(BaseModel):
    """检索增强问答的结构化结果。

    演示场景：用户问 LangChain / LangGraph / 项目本地资料相关问题 →
    agent 调 search_docs → 返回带引用来源的结构化回答。

    设计要点：
      - sources 期望来自检索结果的 source 字段；但模型可能瞎编，所以配合
        verify_rag_sources()（见本文件底部）做「最终来源 ⊆ 实际检索集合」的后校验。
      - confidence < 0.5 时下游可以选择「需要人工二次确认」
    """

    answer: str = Field(description="基于检索内容给出的最终答案")
    sources: Annotated[list[str], Field(min_length=0)] = Field(
        default_factory=list,
        description="引用来源列表（来自检索片段的 source 字段，正斜杠路径）",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="模型对答案的置信度；< 0.5 表示知识库没覆盖、答案可能不准确",
    )


# ────────────────────────────── 工厂函数 ──────────────────────────────


def make_response_format(schema: type[BaseModel]) -> ToolStrategy:
    """把 Pydantic schema 包成 ToolStrategy，喂给 `create_agent(response_format=...)`。

    Args:
        schema: 任意继承自 pydantic.BaseModel 的类。

    Returns:
        ToolStrategy 实例，可直接传 create_agent。

    示例：
        from agent.structured import WeatherReport, make_response_format
        agent = create_agent(
            model=build_llm(),
            tools=[],
            response_format=make_response_format(WeatherReport),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "北京天气？"}]})
        print(result["structured_response"])  # WeatherReport(city='北京', ...)
    """
    return ToolStrategy(schema)


def get_schemas() -> dict[str, type[BaseModel]]:
    """返回本模块所有内置 schema，方便动态注册到 UI / API。

    Returns:
        字典：键是 schema 名称（与类名一致），值是 schema 类。
    """
    return {
        "ChatReply": ChatReply,
        "WeatherReport": WeatherReport,
        "RAGAnswer": RAGAnswer,
    }


def get_default_schema() -> type[BaseModel]:
    """默认 schema：ChatReply（最小可用形态）。"""
    return ChatReply


# ────────────────────────────── 引用来源后校验 ──────────────────────────────
#
# ToolStrategy 只保证「输出形状符合 schema」，不保证字段内容可信——尤其 RAGAnswer.sources
# 模型可以随手编造。verify_rag_sources() 把「这一轮 search_docs 真实返回过的来源路径」当作
# 事实基线，凡是不在基线里的 sources 一律判定为伪造并剔除。


# search_docs 工具输出里的行格式：`[1] 来源: <path>`（见 agent.rag.search_tool）
_SOURCE_LINE_RE = re.compile(r"^\s*\[\d+\]\s*来源:\s*(.+?)\s*$")
# search_docs 命中时的正文标记（用于兜底识别工具消息）
_RETRIEVAL_MARKER = "从本地知识库检索到"


def _norm_source_path(path: str) -> str:
    """来源路径归一化：反斜杠转正斜杠、去首尾空白。"""
    return path.replace("\\", "/").strip()


def _collect_retrieved_sources(result: dict[str, Any]) -> set[str]:
    """从 invoke 返回的消息流里，收集 search_docs 真实返回过的来源路径。

    只认形如 ``[n] 来源: <path>`` 的行（这是工具在片段头部统一打的前缀），
    避免把检索正文里恰好出现「来源:」的文字也误当来源。
    """
    found: set[str] = set()
    for msg in result.get("messages", []):
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # 多模态内容块：拼出 text 部分
            content = "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if not isinstance(content, str) or _RETRIEVAL_MARKER not in content:
            continue
        for line in content.splitlines():
            m = _SOURCE_LINE_RE.match(line)
            if m:
                found.add(_norm_source_path(m.group(1)))
    return found


def verify_rag_sources(
    result: dict[str, Any],
    *,
    schema: type[BaseModel] = RAGAnswer,
) -> tuple[Any, list[str]]:
    """校验结构化输出里的引用来源没被模型伪造，剔除不存在的来源。

    基线 = 本轮 search_docs 工具消息里真正出现过的 ``[n] 来源: <path>`` 集合。
    凡是 ``structured_response.sources`` 里不在该集合内的路径，视为模型编造。

    Args:
        result:  ``build_structured_agent(...).invoke()`` 的返回值
                 （含 ``messages`` 与 ``structured_response`` 两个键）。
        schema:  需要校验的结构化 schema，默认 ``RAGAnswer``（带 sources 字段）。
                 传入不含 sources 的 schema（如 ChatReply）时本函数是 no-op。

    Returns:
        ``(verified, invalid_sources)``：
          - ``verified``：剔除伪造来源后的 schema 实例（副本，不污染原对象）；
            没有伪造时就是原实例。
          - ``invalid_sources``：被判定为伪造、已剔除的来源列表；空 = 全部通过。

    局限说明：基线取自返回消息流里**所有** search_docs 输出（含历史轮次），
    因此「本轮没检索到、但过去某轮检索到过」的同名来源不会被判为伪造。
    """
    resp = result.get("structured_response")
    if resp is None or not isinstance(resp, schema):
        return resp, []

    sources: list[str] = list(getattr(resp, "sources", []) or [])
    if not sources:
        return resp, []

    retrieved = _collect_retrieved_sources(result)
    valid: list[str] = []
    invalid: list[str] = []
    for s in sources:
        (valid if _norm_source_path(s) in retrieved else invalid).append(s)

    if not invalid:
        return resp, []
    verified = resp.model_copy(update={"sources": valid})
    return verified, invalid


# ────────────────────────────── 类型别名 ──────────────────────────────

# 给 type hints 用的统一别名
StructuredResponse = Any  # 实际类型是 schema 的实例；这里用 Any 避免 import 循环
