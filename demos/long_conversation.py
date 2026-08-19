"""假长对话演示 —— 强制触发 SummarizationMiddleware。

目的：在 10~20 轮对话内必然触发摘要，方便观察"消息被压缩"前后的 state 变化。

做法：
    - 把 trigger 阈值压到 500 token（默认是 2000）
    - 把 keep 阈值压到 100 token（默认是 400）
    - 用脚本化对话（不需要人工输入）

观察要点：
    - 每轮打印：消息条数 + token 估算 + 是否触发了 SummaryMiddleware 节点
    - 触发后，下一轮 prompt 里会看到 "SystemMessage" 内容变成摘要
    - 老消息里的具体内容（人名、数字、细节）会被压缩，但仍保留在 state

启动方式：uv run python demos/long_conversation.py
"""

import sys
from pathlib import Path

# 把项目根目录加到 sys.path，便于 demos/ 子目录 import 上层的 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 终端 UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from langchain.agents import create_agent

from agent.context import create_checkpointer, create_store, preference_tools
from agent.llm import build_llm
from agent.middleware import (
    make_human_in_the_loop_middleware,
    make_model_call_limit_middleware,
    make_summarization_middleware,
)
from agent.tools import demo_tools

# 把 threshold 调小，让演示快速触发
DEMO_TRIGGER_TOKENS = 500
DEMO_KEEP_TOKENS = 100

SYSTEM_PROMPT = (
    "你是一个友好、简洁的中文助手。"
    "回答时先给一句话结论，再用要点展开。"
)


# ============ 脚本化对话 ============
# 每条会发给 agent 的用户消息（尽量覆盖多种内容类型，便于看摘要效果）
SCRIPT = [
    "你好，我叫 Alice，是一名数据科学家。",
    "我今年 30 岁，住在上海，喜欢打羽毛球和读科幻小说。",
    "我目前在一家 AI 公司做 RAG 系统，最近在调研 LangChain 的中间件。",
    "请记住我的邮箱：alice@example.com",
    "我的狗叫 Lucky，是一只金毛，今年 5 岁。",
    "我想做一个能自动摘要长对话的 agent，正在学习 SummarizationMiddleware。",
    "你能告诉我 LangChain 的 checkpointer 有哪些实现吗？",
    "我还想加 human-in-the-loop，让敏感工具调用前暂停审批。",
    "顺便问一下，你是什么模型？训练数据截止到什么时候？",
    "好的，现在请把上面的信息总结成一段 100 字内的简介。",
    "谢谢 Alice！再见，下次再聊。",
]


def approx_tokens(messages) -> int:
    """粗略估算 messages 的总 token 数（≈ 字符数 / 3）。"""
    total = 0
    for m in messages:
        c = m.content
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    total += len(str(b.get("text", "")))
        total += 12
    return total // 3


def render(content) -> str:
    if isinstance(content, str):
        return content.replace("\n", " ⏎ ")
    if isinstance(content, list):
        return "".join(
            (b.get("text", "") if isinstance(b, dict) and "text" in b else str(b))
            for b in content
        )
    return str(content)


def main() -> None:
    print("=" * 70)
    print("假长对话演示 —— SummarizationMiddleware 自动触发")
    print("=" * 70)
    print(f"  trigger 阈值：{DEMO_TRIGGER_TOKENS} tokens（默认 2000）")
    print(f"  keep 阈值：  {DEMO_KEEP_TOKENS} tokens（默认 400）")
    print(f"  对话轮数：   {len(SCRIPT)} 轮")
    print()

    # 构造演示用的 agent
    agent = create_agent(
        model=build_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=preference_tools + demo_tools,
        middleware=[
            make_model_call_limit_middleware(),
            make_human_in_the_loop_middleware(),
            make_summarization_middleware(
                trigger_tokens=DEMO_TRIGGER_TOKENS,
                keep_tokens=DEMO_KEEP_TOKENS,
            ),
        ],
        checkpointer=create_checkpointer(),
        store=create_store(),
    )

    thread_id = "demo-long-conversation"
    config = {"configurable": {"thread_id": thread_id}}

    prev_msg_count = 0
    for turn_idx, user_msg in enumerate(SCRIPT, start=1):
        print(f"\n━━━ 第 {turn_idx:02d} 轮 ─── 用户：{user_msg[:50]}…")

        # 同步执行：不用 stream（流式会破坏 demo 输出节奏）
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=config,
        )

        messages = result["messages"]
        last = messages[-1]
        new_msgs = len(messages) - prev_msg_count

        # 检查 state 里是否出现了 "summary" 关键词（说明 SummarizationMiddleware 触发过）
        all_text = " ".join(render(m.content) for m in messages)
        summary_triggered = any(
            kw in all_text
            for kw in ("SESSION INTENT", "## SUMMARY", "上下文", "历史摘要", "AI 摘要")
        ) and any(
            "ConversationSummary" in type(m).__name__
            or "summary" in type(m).__name__.lower()
            for m in messages
        )

        token_est = approx_tokens(messages)

        marker = "⚡ 摘要触发" if summary_triggered else "           "
        print(
            f"  └─ {marker} | 消息数: {len(messages):2d} (+{new_msgs}) | "
            f"token 估算: {token_est:4d} / 触发阈值 {DEMO_TRIGGER_TOKENS}"
        )
        print(f"     AI: {render(last.content)[:80]}…")

        prev_msg_count = len(messages)

    print("\n" + "=" * 70)
    print("演示结束。最终 state 里所有消息：")
    print("=" * 70)
    final_messages = agent.get_state(config).values["messages"]
    for i, m in enumerate(final_messages, 1):
        snippet = render(m.content)[:60]
        print(f"  [{i:02d}] {type(m).__name__:25s} | {snippet}…")


if __name__ == "__main__":
    main()