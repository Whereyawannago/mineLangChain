"""结构化输出 demo —— 让 agent 回复强制收敛到 Pydantic schema。

启动方式：
    uv run python demos/structured_output.py

demo 三段：
  1) ChatReply      最小可用形态（content + confidence），用内置知识直答
  2) WeatherReport  业务集成形态（city / temperature / summary / source）
  3) RAGAnswer      检索增强形态（answer / sources / confidence）—— 演示 search_docs + 结构化

每段都打印：
  - 调用前的入参（schema 类名）
  - 调用后的 result["structured_response"]（不再是字符串，是 schema 实例）
  - 字段值的字段访问演示（report.city / report.confidence 等）

为什么这个 demo 有价值：
  - 把「agent 输出」从「自由文本 + 正则解析」变成「字段直读」，业务代码直接用
  - 对比 main.py：同样的对话，main.py 拿到的是字符串，这里拿到的是 Pydantic 实例
"""

import sys
from pathlib import Path

# 让 demos/ 子目录能找到上层 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 终端 UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from langchain.agents import create_agent  # noqa: E402

from agent import MAX_CONTEXT_TOKENS  # noqa: E402
from agent.builder import build_llm, build_structured_agent  # noqa: E402
from agent.context import create_checkpointer, create_store, preference_tools  # noqa: E402
from agent.middleware import (  # noqa: E402
    make_model_call_limit_middleware,
    make_summarization_middleware,
)
from agent.structured import (  # noqa: E402
    ChatReply,
    RAGAnswer,
    WeatherReport,
    verify_rag_sources,
)


SYSTEM_PROMPT = """\
# 角色
你是一个友好、简洁的中文助手。

# 回答风格
- 先给一句话结论，再用要点展开。
- 避免冗长；不要重复用户已经说过的话。

# 兜底流程（重要）
- 必须按你被指定的结构化 schema 输出，不要输出 schema 之外的字段。
- 如果你不确定或问题超出你的内置知识，直接在 confidence 字段降低分数（如 0.3），
  并在 content / answer 字段如实说明。
- 不要凭空编造 API、配置项、代码细节。
"""


def _bar(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _demo_chat_reply() -> None:
    """最小可用形态：ChatReply（content + confidence）。"""
    _bar("Demo 1: ChatReply —— 最小可用形态")

    agent = build_structured_agent(ChatReply, include_rag=False)

    config = {"configurable": {"thread_id": "demo-structured-chatreply"}}

    user_msg = "你好，请用一句话介绍 LangChain。"
    print(f"  用户：{user_msg}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )

    reply = result["structured_response"]
    print(f"  schema 类型：{type(reply).__name__}")
    print(f"  reply.content    = {reply.content!r}")
    print(f"  reply.confidence = {reply.confidence}")


def _demo_weather_report() -> None:
    """业务集成形态：WeatherReport（city / temperature / summary / source）。"""
    _bar("Demo 2: WeatherReport —— 业务集成形态")

    agent = build_structured_agent(WeatherReport, include_rag=False)

    config = {"configurable": {"thread_id": "demo-structured-weather"}}

    user_msg = "上海今天天气怎么样？给我结构化结果。"
    print(f"  用户：{user_msg}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )

    report = result["structured_response"]
    print(f"  schema 类型：{type(report).__name__}")
    print(f"  report.city        = {report.city!r}")
    print(f"  report.temperature = {report.temperature}")
    print(f"  report.summary     = {report.summary!r}")
    print(f"  report.source      = {report.source!r}")
    print()
    print("  → 业务代码可以直接：")
    print(f"    print(f\"{{report.city}}今天 {{report.temperature}}°C，{{report.summary}}\")")
    print(f"    # → 上海今天 {report.temperature}°C，{report.summary}")


def _demo_rag_answer() -> None:
    """检索增强形态：RAGAnswer（answer / sources / confidence）—— 必须先跑过 ingest。"""
    _bar("Demo 3: RAGAnswer —— 检索增强形态（需要先跑 ingest）")

    try:
        agent = build_structured_agent(RAGAnswer, include_rag=True)
    except FileNotFoundError as e:
        print(f"  ⚠️  跳过：{e}")
        print("  请先跑：uv run python demos/ingest_obsidian_notes.py")
        return

    config = {"configurable": {"thread_id": "demo-structured-rag"}}

    user_msg = "langchain 这个项目里的 SummarizationMiddleware 是怎么用的？"
    print(f"  用户：{user_msg}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )

    # 后校验：剔除模型编造、不在本次检索结果里的引用来源
    rag, fabricated = verify_rag_sources(result)
    print(f"  schema 类型：{type(rag).__name__}")
    print(f"  rag.answer     = {rag.answer[:100]}{'...' if len(rag.answer) > 100 else ''}")
    print(f"  rag.sources    = {rag.sources}")
    print(f"  rag.confidence = {rag.confidence}")

    if fabricated:
        print()
        print(
            f"  ⚠️  检出 {len(fabricated)} 个不在检索结果里的伪造引用，"
            f"已剔除：{fabricated}"
        )
    if rag.confidence < 0.5:
        print()
        print("  ⚠️  模型自评置信度 < 0.5，下游可以选择「需要人工二次确认」")


def _demo_compare_to_free_text() -> None:
    """对比：相同问题在自由文本 vs 结构化输出下的返回值差异。"""
    _bar("Demo 4: 对比 —— 同一个问题，两种 agent 的返回差异")

    from agent import build_agent

    free_agent = build_agent()
    struct_agent = build_structured_agent(ChatReply, include_rag=False)

    user_msg = "1 + 1 等于几？"
    config = {"configurable": {"thread_id": "demo-compare"}}

    print(f"  用户：{user_msg}")
    print()

    free_result = free_agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )
    free_text = free_result["messages"][-1].content
    print(f"  自由文本 agent：")
    print(f"    result['messages'][-1].content  = {free_text!r}")
    print(f"    类型：{type(free_text).__name__}")

    print()

    struct_result = struct_agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
    )
    struct_obj = struct_result["structured_response"]
    print(f"  结构化 agent：")
    print(f"    result['structured_response']    = {struct_obj!r}")
    print(f"    类型：{type(struct_obj).__name__}")
    print(f"    reply.content                   = {struct_obj.content!r}")
    print(f"    reply.confidence                = {struct_obj.confidence}")


def main() -> None:
    print("=" * 70)
    print(" 结构化输出 demo —— ToolStrategy + Pydantic schema")
    print("=" * 70)
    print(f"  摘要触发阈值：{MAX_CONTEXT_TOKENS} tokens")
    print()

    _demo_chat_reply()
    _demo_weather_report()
    _demo_rag_answer()
    _demo_compare_to_free_text()

    print()
    print("=" * 70)
    print(" Demo 结束")
    print("=" * 70)
    print()
    print("关键变化（与 main.py 对比）：")
    print("  - main.py 同一问题 → result['messages'][-1].content 是 str")
    print("  - 这里 → result['structured_response'] 是 Pydantic 实例，可直接 .字段 取值")
    print("  - 业务集成代码不再写正则解析 / JSON 解析")


if __name__ == "__main__":
    main()
